# DFlash2 の受理数に対する入力、sampler、drafter の寄与

Qwen3.8-27B、chain 幅 8 の [Issue #44](https://github.com/kusuriya-labs/shelf/issues/44) を調べた。受理数は target が採用した draft の個数で、anchor・修正 token・bonus は含めない。

同一条件の 9 要求で、sampler だけを変更した比較では、前後とも 3282 round・10232 draft 受理（3.118 個/round）で、生成した 13523 個の ID がすべて一致した。AB/BA の合算速度は 99.59 → 99.98 tok/s。各要求の生成数から 1 を引いた合計を decode 時間の合計で割った。sampler 単体には短縮があり、今回の seed で受理数への寄与は 0 個だった。

最終版では GDN の融合演算が参照する入力の生存期間も修正した。同じ seed で 8 問の生成列が変わった。9 問の AB/BA 合算は 3.118 → 3.105 受理/round、100.42 → 98.99 tok/s。

## 比較条件

基準は shelf `dffb8851`、sampler 単独修正版と、その上に GDN の生存期間修正を加えた最終版を測定した。R9700、ROCm 10.2、gfx1201、Release、TP=1、単一要求で全実行を直列化した。GPU は `ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55`、排他は `flock /tmp/r9700.lock`。shelf は `qwen38-27b-exl3-305-mul24` target、`dflash2-drafter-exl3-4bpw-mul24` drafter、BF16 KV、5 層の drafter、drafts 7、tree budget 0、prefix cache 有効、最大系列長 24576、prefill chunk 2048。元 BF16 drafter は `Qwen3.8-27B-DFlash2`。saber は `e43b8877` の MACAQ W4A8 target と W4A8-iu8 drafter、BF16 の共有 head、FP8 の drafter KV、最大系列長 32768。各サーバーの既定の停止 token を使った。

real-v1 は `code_edit-{ja,en}-{short_2k,mid_8k,long_16k}` の 6 問、生成上限 2048。BetterBench v1 は各カテゴリの先頭の JSON 抽出・コード生成・水槽の文章題の計 3 問で、上限は順に 300・600・500。nonce を追加せず、温度 0.7、top-k 20、top-p 0.95、min-p 0、seed 42 に固定した。9 問とも thinking を有効にした。元の BetterBench 集計は複数問、nonce あり、seed 未指定であり、今回の各 1 問とは母集団が異なる。

shelf の tokenizer と chat template で各入力を一度だけレンダリングし、shelf stdio の `input_ids` と saber の `/v1/completions` の整数配列 `prompt` に渡した。全 9 問で xhigh の先頭 42 ID を除くと medium と同一だった。9 問 × 2 条件の全 ID が server の `codec.prompt_ids` と一致した。medium と強制終了なしが共通条件。要求ごとに shelf は `DFPHASE_REQUEST` の実受理数と round 数、saber は前後の `spec_decode_num_{drafts,accepted_tokens}_total` の差分を使った。saber のカウンタは最終ブロックの生成上限を超えた検証も含みうる。速度は生成数から 1 を引き、最初と最後の token 通知の時間差で割った。shelf は stdio、saber は HTTP SSE で計時した。

## 同一入力での実受理数と速度

次表では sampler のみを変更した。shelf は各プロセスを短い JSON で暖機し、基準 A と候補 B を AB、BA の順で測定した。表の速度は 2 回の平均。AB は通常の round trace、BA は両版で Graph 起動診断も有効にした。saber の速度は同じ入力・暖機で 1 回。Graph 診断は別の実行で取り、9 要求の生成 ID の再現を確認した。各試行の ID ハッシュ、思考終了位置、生成数は [要求別 CSV](issue44_dflash2_acceptance/runs.csv) に収録した。入力・出力 ID のハッシュはリトルエンディアン uint32 列の SHA-256。CSV の `native` は旧既定、`template` は xhigh・予約なし。`oldA/newB/newC/oldD` は sampler 単独の AB/BA、`finalA/finalB/finalC/finalD` は基準と最終版の AB/BA、`saber-replay` は saber の Graph 診断試行を表す。

|課題|入力 ID 数|shelf 旧 round / 受理|shelf 新 round / 受理|旧 → 新 受理/round|旧 → 新 tok/s|saber round / 受理|saber 受理/round|saber tok/s|
|---|---|---|---|---|---|---|---|---|
|日 2k|2071|426 / 1621|426 / 1621|3.81 → 3.81|111.8 → 109.3|481 / 1568|3.26|118.4|
|日 8k|8028|610 / 1437|610 / 1437|2.36 → 2.36|81.5 → 82.7|609 / 1440|2.36|92.4|
|日 16k|16390|500 / 1547|500 / 1547|3.09 → 3.09|96.4 → 97.6|514 / 1533|2.98|107.5|
|英 2k|2041|450 / 1597|450 / 1597|3.55 → 3.55|114.0 → 114.6|431 / 1617|3.75|132.0|
|英 8k|7989|591 / 1456|591 / 1456|2.46 → 2.46|85.0 → 85.1|636 / 1411|2.22|88.6|
|英 16k|16366|483 / 1564|483 / 1564|3.24 → 3.24|100.5 → 101.1|505 / 1542|3.05|107.6|
|JSON|63|25 / 109|25 / 109|4.36 → 4.36|128.3 → 133.9|26 / 136|5.23|178.5|
|code|56|107 / 492|107 / 492|4.60 → 4.60|146.8 → 147.8|115 / 488|4.24|150.2|
|math|60|90 / 409|90 / 409|4.54 → 4.54|143.4 → 144.0|94 / 405|4.31|152.9|

共通条件の要求ごとに成功した hipGraphLaunch を数えた。次表は初回実行も含む全 Graph の起動回数。shelf では target verify の 8 行 cache hit も別に数えた。両エンジンで Graph の分割単位が異なる。

|課題|shelf 全 Graph|うち 8 行 verify 再生|saber 全 Graph|
|---|---|---|---|
|日 2k|850|424|963|
|日 8k|1219|609|1219|
|日 16k|995|497|1029|
|英 2k|895|447|863|
|英 8k|1180|590|1273|
|英 16k|963|481|1011|
|JSON|50|24|53|
|code|214|107|231|
|math|179|89|189|

## GDN 修正を含む最終版の AB/BA

基準・最終版を改めて AB、BA の順に実行した。次表では各要求の 2 回の速度の平均を示す。受理数は繰り返し間で一致した。生成文が変わるため、同じ round の処理時間だけを比較した値とは異なる。

|課題|基準 round / 受理|最終 round / 受理|受理/round|decode tok/s|8 行 round p50 ms|最終 Graph 起動|最終 8 行 verify 再生|
|---|---|---|---|---|---|---|---|
|日 2k|426 / 1621|467 / 1580|3.805 → 3.383|120.9 → 96.6|39.51 → 38.84|932|465|
|日 8k|610 / 1437|568 / 1479|2.356 → 2.604|81.0 → 89.3|40.68 → 40.24|1134|567|
|日 16k|500 / 1547|489 / 1558|3.094 → 3.186|97.1 → 99.1|42.01 → 41.65|976|488|
|英 2k|450 / 1597|380 / 1667|3.549 → 4.387|114.0 → 135.4|39.77 → 39.44|757|378|
|英 8k|591 / 1456|610 / 1437|2.464 → 2.356|84.4 → 82.7|40.61 → 40.30|1217|608|
|英 16k|483 / 1564|563 / 1484|3.238 → 2.636|100.5 → 86.4|42.02 → 41.63|1119|559|
|JSON|25 / 109|25 / 109|4.360 → 4.360|133.4 → 134.1|37.33 → 37.02|50|24|
|code|107 / 492|108 / 491|4.598 → 4.546|141.8 → 146.7|37.91 → 37.58|214|107|
|math|90 / 409|82 / 417|4.544 → 5.085|143.5 → 161.2|37.72 → 37.42|162|81|

8 行 round は同じ要求の直前の round 終了から 7 提案の round 終了までの時間で、各試行の p50 を 2 回で平均した。全 9 問でこの p50 は短縮した。日本語 2k の最終版には 3461 ms と 2444 ms の間隔がそれぞれ 1 回あり、基準の 2 試行では最長 43 ms だった。tok/s の集計にはこれらの間隔も含めた。

## 条件差の内訳

各行で同じ seed の要求を順に比較した。①旧版の xhigh・回答用予約あり、②同じ入力で予約を外す、③medium にして追加指示を外す、④sampler を変更、⑤GDN の入力の生存期間を修正、⑥saber に交換、の順。差分はこの順序で観測した受理数/round の変化。生成文が分岐するので、別の順序や seed にそのまま加算できる効果量ではない。最後の差には target・drafter の重みと数値演算、乱数、同点の扱い、末尾のカウント規則がともに入る。

|課題|① 既定|② 予約なし|③ 共通入力|④ 新 sampler|⑤ 最終版|⑥ saber|予約差 ②−①|指示差 ③−②|sampler ④−③|GDN ⑤−④|残差 ⑥−⑤|
|---|---|---|---|---|---|---|---|---|---|---|---|
|日 2k|2.024|1.920|3.805|3.805|3.383|3.260|-0.104|+1.885|+0.000|-0.422|-0.123|
|日 8k|1.546|1.454|2.356|2.356|2.604|2.365|-0.092|+0.901|+0.000|+0.248|-0.239|
|日 16k|1.708|1.478|3.094|3.094|3.186|2.982|-0.229|+1.616|+0.000|+0.092|-0.204|
|英 2k|2.169|1.895|3.549|3.549|4.387|3.752|-0.273|+1.654|+0.000|+0.838|-0.635|
|英 8k|1.565|1.364|2.464|2.464|2.356|2.219|-0.201|+1.100|+0.000|-0.108|-0.137|
|英 16k|1.984|1.628|3.238|3.238|2.636|3.053|-0.356|+1.610|+0.000|-0.602|+0.418|
|JSON|3.706|3.706|4.360|4.360|4.360|5.231|+0.000|+0.654|+0.000|+0.000|+0.871|
|code|3.608|3.608|4.598|4.598|4.546|4.243|+0.000|+0.990|+0.000|-0.052|-0.303|
|math|4.366|4.366|4.544|4.544|5.085|4.309|+0.000|+0.179|+0.000|+0.541|-0.777|

既定の real-v1 は全 6 問で生成 ID の 1537 番目が思考終了 token だった。これを含む各 1 round を除くと、終了前は 3507 round で 5701 受理（1.626 個/round）、終了後は 854 round で 2212 受理（2.590 個/round）。同じ出力内でも回答部分の受理率は思考部分より高かった。共通条件では基準・sampler 単独修正版と saber が全 6 問で上限まで思考を継続した。最終版は日本語 8k のみ 1857 個目で自然に思考を閉じた。

6 問の全 round を合算した比率では、既定 1.812 → xhigh・予約なし 1.606 → medium・予約なし 3.014 → 新 sampler 3.014 → 最終版 2.992 → saber 2.869 個/round。差は順に -0.206、+1.408、+0.000、-0.022、-0.123 個/round。

## 実ステップの target logits と乱数位置

共通条件で sampler 単独修正版が検証した幅 8 の 8 round ごとに 1 回、先頭 7 行の F32 logits 全 248320 語、提案 ID、target が引いた ID、sample index を保存した。暖機は除外。各行の同じ logits に旧・新の両閾値を適用し、残る語だけを再正規化した。greedy selector の提案分布 q は一点分布なので、位置ごとの理論受理確率 Σ min(p,q) は p(draft) に等しい。先頭から連続して受理する長さ L の期待値は、提案経路に沿った各位置の確率の累積積を 7 位置で足した値。ここでは幅 8 の verify における比較そのものを測り、EOS や生成上限での打ち切りは適用していない。たとえば日本語 2k の sample index 679・提案位置 7 では、質量が 3.066465 → 2.979264、候補数が 13 → 9 になった。候補数が変わった全 20 行の ID・logits・分母は [境界行の記録](issue44_dflash2_acceptance/sampler_differences.jsonl) に収録した。

|課題|round|候補数が変化した行|平均候補数 旧 → 新|平均 p(draft) 旧 → 新|E[L] 旧 → 新|受理合計 seed 42|
|---|---|---|---|---|---|---|
|日 2k|53|3|1.4771 → 1.4609|0.758778 → 0.758837|3.910287 → 3.910287|202 → 202|
|日 8k|76|7|1.8195 → 1.7575|0.612673 → 0.612697|2.868531 → 2.868814|209 → 209|
|日 16k|62|4|1.6106 → 1.6014|0.671507 → 0.671446|3.132456 → 3.132456|180 → 180|
|英 2k|56|0|1.4515 → 1.4515|0.752535 → 0.752535|3.615949 → 3.615949|193 → 193|
|英 8k|74|5|1.8031 → 1.7838|0.631494 → 0.631501|2.209817 → 2.209817|181 → 181|
|英 16k|60|1|1.5881 → 1.5714|0.667056 → 0.667056|2.936404 → 2.936404|191 → 191|
|JSON|3|0|1.0476 → 1.0476|0.960006 → 0.960006|6.440088 → 6.440088|19 → 19|
|code|14|0|1.1633 → 1.1633|0.886775 → 0.886775|5.103636 → 5.103636|77 → 77|
|math|11|0|1.1948 → 1.1948|0.849450 → 0.849450|4.655067 → 4.655067|48 → 48|
|全体|409|20|1.6112 → 1.5903|0.689356 → 0.689360|3.187400 → 3.187453|1300 → 1300|

元の sample index、token ID、seed 42 から Gumbel-Max を再生し、新方式で GPU の実 draw が全 2863 行一致した。同じ logits に seed 0〜255 を与えた平均受理長は旧 3.188360、新 3.188360。新方式で乱数位置だけを確定 token 数に替えると seed 42 の受理合計は 1300 → 1312、256 seed 平均は 3.188360 → 3.187443。物理 verify 行数を進める現行契約は両経路で一貫し、固定 logits 上の理論確率は乱数位置に依存しない。この比較から受理率向上を目的とする乱数位置の変更は採用しなかった。

sampler 単体の所要時間も同じ GPU で測った。`shelf-bench sampling --variant stochastic-radix --vocab 248320 --logit-pattern unique --temperature 0.7 --top-k 20 --top-p 0.95 --seed 42 --warmup 100 --samples 31 --launches 10`。出力行数だけ 1 と 8 に替え、AB / BA を交互に各 5 回測った。表は HIP event の試行別 p50 の中央値（最小–最大）。全場合で候補は 1 語。打ち切りの分母に使う exp の計算が 248320 語から 20 語になった効果を含む。

|行数|旧 µs|新 µs|短縮率|
|---|---|---|---|
|1|585.3 (583.8–585.4)|353.6 (353.1–353.8)|39.6%|
|8|540.8 (539.6–544.5)|357.2 (356.3–357.4)|33.9%|

## 共通 prefix における位置別の監査

各課題の共通入力に saber の同じ出力列を 0・8・24・48・96・160 token だけ付けた 54 prefix を用意した。各 prefix から greedy prefill で anchor を生成し、最初の 7 提案と、各提案経路に沿って target が返す argmax・確率を取得した。EXL3 と元 BF16（`Qwen3.8-27B-DFlash2`）は同じ shelf target と head を使用。saber では greedy の verify 行から上位 20 語を読み、同じ温度と打ち切り定義で p(draft) を算出した。saber 実 sampler は k 位の同点をすべて残し、shelf は token ID 順に k 個に絞る。位置別の表は両者に同じ k 個の規則を使った。

|drafter / 条件|1|2|3|4|5|6|7|greedy 受理長|E[L]|
|---|---|---|---|---|---|---|---|---|---|
|shelf EXL3|92.6%|88.9%|87.0%|77.8%|75.9%|57.4%|72.2%|4.444|4.282|
|shelf 元 BF16|90.7%|90.7%|85.2%|77.8%|77.8%|57.4%|68.5%|4.352|4.205|
|shelf 最終 EXL3|92.6%|92.6%|88.9%|77.8%|77.8%|57.4%|74.1%|4.519|4.253|
|shelf 最終 元 BF16|92.6%|92.6%|88.9%|77.8%|79.6%|59.3%|74.1%|4.500|4.249|
|saber W4A8|88.9%|83.3%|79.6%|68.5%|72.2%|53.7%|51.9%|3.944|3.878|
|shelf tap を 1 層前へ|92.6%|90.7%|87.0%|81.5%|72.2%|59.3%|63.0%|4.333|4.246|
|shelf 全語彙の元の head|92.6%|88.9%|87.0%|77.8%|75.9%|57.4%|72.2%|4.444|4.282|
|shelf EXL3 を BF16 に展開|92.6%|88.9%|87.0%|77.8%|77.8%|57.4%|72.2%|4.444|4.282|
|shelf prefix cache 無効|90.7%|88.9%|87.0%|77.8%|75.9%|59.3%|70.4%|4.333|4.194|

最終版の EXL3 と元 BF16 の greedy 受理長は 4.519 と 4.500、E[L] は 4.253 と 4.249 だった。この 54 prefix で、EXL3 化による平均受理長の低下は観測しなかった。

次は同じ 54 prefix で、位置ごとの target の p(draft) を平均した値。最初の棄却より後も、既に提案した token を条件にその行を評価した。

|drafter|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|
|shelf EXL3|0.919|0.857|0.853|0.777|0.766|0.583|0.701|
|shelf 元 BF16|0.917|0.867|0.835|0.777|0.779|0.565|0.682|
|shelf 最終 EXL3|0.920|0.873|0.849|0.773|0.768|0.571|0.715|
|shelf 最終 元 BF16|0.920|0.877|0.849|0.775|0.787|0.585|0.707|
|saber W4A8|0.898|0.810|0.804|0.702|0.720|0.536|0.510|

課題別に 6 prefix を平均した。各欄は greedy の実受理長 / 同じ提案を sampled verify した場合の理論受理長。

|課題|shelf EXL3|shelf 元 BF16|saber W4A8|
|---|---|---|---|
|日 2k|3.17 / 3.05|3.00 / 2.96|2.00 / 2.49|
|日 8k|2.83 / 2.32|3.00 / 2.42|2.00 / 1.71|
|日 16k|4.33 / 4.14|4.33 / 4.14|3.67 / 3.57|
|英 2k|4.67 / 4.40|4.67 / 4.40|4.17 / 4.05|
|英 8k|4.50 / 4.44|4.50 / 4.44|3.67 / 3.65|
|英 16k|4.00 / 3.74|3.17 / 3.02|3.00 / 2.80|
|JSON|5.17 / 5.09|5.17 / 5.09|5.17 / 5.17|
|code|5.33 / 5.62|5.33 / 5.62|6.67 / 6.24|
|math|6.00 / 5.74|6.00 / 5.74|5.17 / 5.21|

次は target の判定を介さず、同じ prefix・同じ anchor に対する 2 つの drafter の提案 ID を直接照合した値。

|提案の比較|prefix 数|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|---|
|基準 EXL3 と元 BF16|54|98.1%|96.3%|96.3%|96.3%|94.4%|85.2%|81.5%|
|最終 EXL3 と元 BF16|54|100.0%|96.3%|98.1%|96.3%|96.3%|90.7%|87.0%|
|最終 EXL3 と saber W4A8|52|92.3%|88.5%|90.4%|73.1%|61.5%|61.5%|50.0%|

anchor が 3 条件で一致した 52 prefix では、saber の W4A8 が出した 7 個の提案 ID を shelf の verify 入力へそのまま渡して全行の logits を再取得した。E[L] は shelf 提案・shelf target が 4.4102、saber 提案・shelf target が 4.0662、同じ saber 提案・saber target が 3.9730。shelf target を固定して提案列だけを交換した差は -0.3441、提案列を固定して target の検証分布を交換した差は -0.0931。前者には drafter が元のエンジンから受け取った tap と head の効果も含む。固定 prefix における期待値であり、自由生成中の要求平均とは分母が異なる。

最終版でも同じ交換を繰り返した。共通 anchor の 52 prefix で E[L] は 4.3308 → 4.0690 → 3.9730。同じ shelf target 上の提案列の寄与は -0.2618、同じ saber 提案上の target 検証分布の寄与は -0.0959。

さらに target-only の shelf greedy 継続を別に生成した。次表はその継続の各位置との提案 ID 一致率であり、途中で提案が外れても比較対象の継続は固定した。anchor が一致し、停止までに少なくとも 7 語の継続がある prefix に限った。

|drafter|prefix 数|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|---|
|shelf EXL3|53|92.5%|86.8%|83.0%|67.9%|54.7%|45.3%|49.1%|
|shelf 元 BF16|53|90.6%|88.7%|79.2%|67.9%|54.7%|45.3%|49.1%|
|shelf 最終 EXL3|53|92.5%|86.8%|81.1%|67.9%|58.5%|45.3%|50.9%|
|shelf 最終 元 BF16|53|92.5%|86.8%|81.1%|67.9%|62.3%|45.3%|50.9%|
|saber W4A8|51|88.2%|86.3%|80.4%|60.8%|51.0%|39.2%|35.3%|

通常の EXL3 の提案 ID との一致数は、1 層前の tapでは 312/378、全語彙の元の headでは 378/378、展開した EXL3 重みでは 376/378、prefix cache 無効では 373/378。cache 無効は長文を先頭から prefill するため、復元の有無と prefill の行分割の両方が変わる。cache 有効時に系統的に受理が悪化する傾向はこの比較では見られなかった。

各 prefix の生成済み ID、ハッシュ、anchor、全提案 ID、verify argmax、p(draft)、target-only 継続は [位置別の観測値](issue44_dflash2_acceptance/prefixes.jsonl) に収録した。

## 実装の判定と検証

CLI の checkpoint 数のヘルプを実際の既定 16 に合わせ、prefill chunk の既定 4096 を転送することを CLI テストで確認した。長文 prefix の回帰テストが前提とする 2048 token 間隔の checkpoint をテスト内で明示した。製品既定の保存間隔 16384 ではこのテストの 4300 token の入力に中間 checkpoint ができない。8 項目が通過し、cache から復元した出力は cache 無効の DFlash2 と非投機 greedy の両方と一致した。

単発 CLI の block size 8 では、stdio と異なる feature 作業行数と prefill 分割で target tap を処理していた。元から block size 16 が使用していた共通の feature 容量判定と、最終ページ手前の分割を block size 8 にも適用した。39 token の同じ入力で budget 10 の単発と HTTP の生成列が異なる回帰を修正し、chain・budget 8・10・auto のテストが通過した。上の速度と受理数は stdio 経路で測った。単体 CLI の既定 tree budget のヘルプを実装の 0 に合わせ、auto の動作を調べるテストでは明示指定した。

sampler の旧動作は文書化された積集合だった。比較先と同じ順序で top-k と nucleus を指定できるよう、残った top-k の質量を分母にする契約へ変更した。通常の radix、correctness backend、混在行や k > 128 の rejection、DFlash2 sampled verify に適用する。greedy と min-p の閾値、同点で token ID を優先する規則を維持した。回答用予約は要求単位で解除でき、Track C は思考指示と予約を独立に設定できる。

shelf は宣言された層の出力残差を読み、saber の比較版は層の入力残差を読む。SGLang の Qwen3.5 テキストモデルは指定層に 1 を足し、次層で直前の出力を捕捉する。shelf の tap を 1 層前へずらした対照実験は上表に示した。block は anchor と mask 7 個、RoPE は絶対位置、非 causal な両側窓 2048、selector は先行提案を次位置へ渡す。確定行だけを ring に追加し、GDN 状態は受理した行に戻す。各機構の独立した速度・受理差はこの比較からは算出できない。[SGLang の捕捉位置](https://github.com/sgl-project/sglang/blob/2c5b680aa6/python/sglang/srt/models/qwen3_5_text.py)・[saber の tap](https://github.com/GGZ14/saber-engine/blob/e43b887776a6a37ddd591a2628ce6c4c14ca574e/crates/saber-engine/src/aux_tap.rs)

GDN を変更する前の 3.05 bpw の `test_compute_dflash2_unaligned.py` は 43 項目中 41 項目が通過した。非投機と投機の 256 ID の完全一致は cache の有無の両方で失敗した。基準と sampler 単独修正版で実行し、非投機の列同士、および投機の列同士はいずれも 256 ID すべて一致した。分岐は両版とも最初の 115 token が一致した後の 116 個目で、非投機が 13990、投機が 264。非投機の top-1 と top-2 の logit 差は 0.038290。verify の当該 2 語の logits は 20.541054 と 20.609791 で順位が逆転した。同じ stdio セッションの全 round で提案数を 0 に固定し、DFlash2 の既存の単一 target decode だけを通した対照でも、初回差は同じ 116 個目の 264 だった。115 個の生成まで単一 target decode を実行し、その直後だけ通常の 8 行 verify に戻した場合も同じ 2 語の logits が得られた。提案や棄却後の GDN 巻き戻しなしでこの差を再現できる。このテストは greedy で sampler の打ち切りを通らない。両側で GDN recurrence の exact 指定も試したが、初回差は 61 個目となった。同じ通常 CLI に 5 層の hidden tap だけを追加すると投機側の生成列と 256 ID がすべて一致した。[同じ入力での過去の測定](../20261122/dflash2_phase2/README.md)

rocprofv3 で同一の非投機 CLI を測ると、104 token の入力を 96 行・8 行に分けた prefill と次の 1 行 decode のいずれも、GDN の RMSNorm と gate を融合できたのは 48 層中、tap なしで 1 層、tap ありで 15 層だった。tap を external output として確保すると他の作業領域の再利用位置も変わる。融合演算は recurrence 出力と SiLU 前の gate 入力を乗算時に直接読むが、非融合の生存期間ではその場所を既に乗算出力へ再利用できた。安全検査は重なる層で非融合を選ぶため、tap の有無だけで計算経路が変わった。GDN の融合を両側で解除すると 256 ID が一致し、residual・次層の射影の融合だけを解除しても 116 個目の差は残った。

融合カーネルの二乗和の順序だけを合わせる診断版は 8 個目で分岐し、不採用とした。最終版では、該当する EXL3・6144 幅・128 要素/head の SiLU gate のグラフで、recurrence 出力と gate 入力を乗算まで生存させた。これにより 96・8・1 行のすべてで、tap の有無にかかわらず 48/48 層で融合を使った。同じ入力の非投機 greedy と投機の 256 ID、cache 復元位置および生成列を含む `test_compute_dflash2_unaligned.py` は 43/43 通過した。許容誤差とアサーションは変更していない。

Graph は target verify と drafter proposal の両方を ON/OFF し、9 課題 × greedy / sampled の 18 要求、生成上限 64、1152 token で照合した。全生成 ID、各 round の位置・sample index・提案 ID、要求別 round 数・実受理数が一致した。ROCm 10.2 Release の全体ビルドと required CTest 224/224 が成功し、skip は 0 件。実モデルの server DFlash2 は XGrammar なしで 20/20、ありで 21/21、予算テストは 28/28、reasoning request は 49/49、ベンチスイートは 14/14。実サーバーへ Track C の英語 16k を `--reasoning-effort medium --no-answer-reserve` で送っても、usage は入力 16366・出力 2048、回答欄は空、sampler 単独修正版で 483 round・1564 受理、最終版で 563 round・1484 受理となり、それぞれ stdio の共通条件を再現した。

最終版のサーバー回帰は `core`・`reasoning`・`bench` の計 33 スクリプトが通過した。PLaMo tokenizer のテストは共通の Python 環境に numba がなかったため、対応環境で再実行して 6/6 を確認した。単一 GPU 指定のため、thinking budget スクリプト内の TP=2 専用ケースは対象外。単発 tree は 3 入力・各 256 token の全 budget で既存の判定基準を通過した。終了時の KFD プロセス数は 0。

## 変更ファイル

実装・ベンチ　[sampling_common.h](../../../include/shelf/kernels/optimized/sampling_common.h)、[gdn_lowering.h](../../../include/shelf/models/common/qwen_hybrid/gdn_lowering.h)、[compute_cli.cpp](../../../src/apps/compute/compute_cli.cpp)、[main.hip](../../../src/apps/compute/main.hip)、[single_request.hip](../../../src/apps/compute/single_request.hip)、[chat_service.py](../../../src/apps/server/serverlib/chat_service.py)、[openai_protocol.py](../../../src/apps/server/serverlib/openai_protocol.py)、[shelf_server.py](../../../src/apps/server/shelf_server.py)、[shelf_bench_suite.py](../../../tools/bench/shelf_bench_suite.py)

テスト　[test_stochastic_sampling.hip](../../../tests/kernels/optimized/test_stochastic_sampling.hip)、[test_stochastic_topk.hip](../../../tests/kernels/optimized/test_stochastic_topk.hip)、[test_compute_dflash2_prefix_long.py](../../../tests/server/test_compute_dflash2_prefix_long.py)、[test_inference_budget.py](../../../tests/server/test_inference_budget.py)、[test_server_cli.py](../../../tests/server/test_server_cli.py)、[test_server_dflash2_switch.py](../../../tests/server/test_server_dflash2_switch.py)、[test_server_dflash2_tree.py](../../../tests/server/test_server_dflash2_tree.py)、[test_shelf_bench_suite.py](../../../tests/server/test_shelf_bench_suite.py)

契約・操作説明　[dflash2.md](../../developer/dflash2.md)、[reasoning.md](../../developer/reasoning.md)、[sampling.md](../../developer/sampling.md)、[bench.md](../../user/bench.md)、[compute.md](../../user/compute.md)、[server.md](../../user/server.md)

計測記録　[issue44_dflash2_acceptance.md](issue44_dflash2_acceptance.md)、[prefixes.jsonl](issue44_dflash2_acceptance/prefixes.jsonl)、[runs.csv](issue44_dflash2_acceptance/runs.csv)、[sampler_differences.jsonl](issue44_dflash2_acceptance/sampler_differences.jsonl)
