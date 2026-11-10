# DFlash2 の draft tree 調査

## 用語と文献

ここでは 1 回の block diffusion forward から得た候補を、anchor を根とする 1 本の prefix-closed な木として検証する。複数 session を同時に処理するときは、その独立した木の集合が forest になる。1 session で互いに prefix を共有しない複数の draft を別々に検証する方式ではない。

- [DDTree](https://arxiv.org/abs/2604.12989) は位置ごとの draft 確率の積を到達確率の代理値とし、固定 node 予算内の候補を best-first で選ぶ。祖先だけを見せる attention で一括検証し、target が選んだ token と一致する子をたどる。
- [CaDDTree](https://arxiv.org/abs/2606.01813) は受理数だけではなく、draft の固定費と木の大きさ・文脈長に依存する verify 時間で割った throughput を目的とする。予算の決定には実機の verify latency の測定が必要である。
- [TAPS](https://arxiv.org/abs/2606.00487) は marginal と分岐後の条件付き受理率のずれを補正する学習済み scorer で prefix-closed な木を選ぶ。学習とその検証が必要である。
- [DForkTree](https://arxiv.org/abs/2611.12356) は fork 後の marginal の精度低下に着目し、target 出力から収集した n-gram の PMI で分岐後の score を補正する。corpus の準備と機器別の検証が必要である。

最初の候補は追加学習や corpus の要らない DDTree 系とし、検証予算は R9700 の低バッチで計測して決める。計測前に CaDDTree の最適予算を仮定しない。DForkTree と TAPS の指摘から、分岐後も同じ位置の marginal を使う score はあくまで代理値である。

## 既存機構と不足

- `ExecuteBatchOptions` は行ごとの祖先位置、確定済み prefix の終端、親行、RoPE 位置を受け取る。paged attention は確定 prefix と祖先のみを参照できる。木の物理 KV 行は depth とは異なる位置に置き、RoPE には depth を使う必要がある。
- GDN recurrence と conv は `tree_row_parent` から分岐できるが、親行の recurrent と conv history が必要である。現在の DFlash2 batch context は conv history のみを確保し、recurrent state は compact log を用いる。tree の入力にそのまま流すと dispatch の検査により拒否される。全行 recurrent history の VRAM と転送量を見積もる必要がある。DForkTree は別の祖先マスク付き chunk 計算と compact な中間量でこのコストを避けているが、shelf の現行 tree kernel がその方式を実装しているわけではない。
- `commit_gdn_compact_log` は任意の行番号列を受け取れるが、現在の batch commit は連続 prefix を replay する。tree では選択経路の行番号を渡し、conv history は選択した最終行から復元する必要がある。代替として最終行の recurrent history からの復元も考えられるが、数値上の差と性能を検証する。
- 選択経路の KV は物理的に連続しないため、採用行を連続位置へ安全に詰め直す必要がある。target tap も同じ経路を集めて drafter context に追加する。単に sequence position を戻すだけでは別枝の KV と tap が混入する。
- DFlash2 selector は通常の位置別 top-k logits だけでなく、直前 token と codebook に基づく補正を加えて chain を逐次選ぶ。既存 selector の条件付き score を別枝にも使うなら、その枝の親 token に対する再計算が必要である。marginal を使う DDTree 自体には必須ではない。DForkTree の DFlash2 オフライン評価は、top-k 内だけで正規化すると残余確率を失い、分岐後の score を過大評価すると報告している。
- 現在の verify は各行の target 出力を sample し、draft と一致する間を受理する。確率的経路でも選択済み prefix の各 target 分布から sample する契約を維持する。行番号と深さが異なると sampling index の対応が変わるため、seed の扱いと履歴依存 sampling を確認する。
- 現在のバッチ行数、target scratch、GDN history、ページ境界は chain の `K+1` 行を前提に確保される。木の予算に応じた容量、複数 session の parent と ancestor の行オフセット、ページ境界、EOS・残り出力長を合わせて扱う必要がある。

## 現状

M1b は単一 request の `shelf-compute` に接続した。既定は従来の chain で、
`--dflash2-tree-extra` を明示した場合だけ木として検証する。chain と最大２個の
兄弟葉を検証し、選択経路の KV・tap を gather する。採用した葉は連続位置で
１行再実行して状態と補正 token を確定する。greedy の 256 token 出力はコマンド、
コード、英語の推論の３ prompt について B=0,1,2,4 と非投機 target で一致した。

## 分岐ごとの候補 score の準備

candidate selector は従来、1 行目の anchor 以降を直前行の選択結果で逐次採点する。分岐先ではその直前行が親とは限らないため、行ごとに GPU 上の親 token を指定して独立に採点できる経路を追加した。各行を別 block で処理し、同じ draft 位置を複数の親で採点できる。既存の batch request 経路とは同時に指定できない。親 token と候補 token の範囲は呼び出し側が保証する。

GPU 単体テストでは同じ候補列について親 3 と親 5 を指定し、それぞれ異なる候補が選ばれ、各候補の score が CPU 参照に一致することを確認した。既存の chain も同じテストで確認した。この selector を単一 request の枝候補選択に接続した。確率値への正規化（top-k 外の残余確率を含む）は実装していない。

M2 では serve 経路への接続、DDTree の予算付き候補選択、確率的検証、
KV gather と葉の再実行の最適化を扱う。枝ありの確率 sampling は現在拒否する。

## 数値差の調査

コード prompt で枝葉の状態をそのまま復元すると、同一 token 列でも position 153 の
verify 行 3 の greedy token が異なった。検証前の GDN recurrent state は最初に
GDN index 9（モデル layer 12）で差があり、layer 5 の tap は一致、layer 19 の tap は
不一致だった。連続しない祖先は attention の別の累算経路を通る。現在は採用葉を
連続位置で再実行し、補正 token・KV・状態・tap をその結果から確定する。再実行を
省いて同じ出力を保つ方法は未検証であり、M2 で扱う。

## M1b の測定

R9700、Qwen3.8-27B EXL3 4bpw mul24、Qwen3.8-27B-DFlash2、BF16 KV、
`temperature=0`、`--dflash2-drafts 7`、生成 256 token。コマンドの長い解説を
求める同じ prompt を各 B について２回実行した。受理長は受理した draft token 数を
round 数で割り、round 時間は `DFLASH2_ROUND_MS / DFLASH2_ROUNDS` とした。
測定には枝葉を採用した場合の１行再実行と同期を含む。速度の優劣は判断しない。

| B | 受理長（２回） | round 時間 ms（２回） |
| --- | --- | --- |
| 0 | 3.048, 3.048 | 55.255, 55.372 |
| 1 | 3.474, 3.474 | 62.611, 67.264 |
| 2 | 3.474, 3.474 | 65.658, 65.758 |
| 4 | 3.474, 3.474 | 65.715, 65.726 |

先頭２位置について各１候補だけを追加するため、B=4 の実際の追加葉は最大２個である。
測定は `tests/server/test_compute_dflash2_tree.py` の最初の prompt の２回実行で取得した。

## M2 の途中計測: 既存の木の内訳

上記と同じモデル、生成長、greedy 条件で、３ prompt を各３回、chain、B=0、B=1 の順に繰り返した。計時用に候補選択、accept walk、木の確定と葉の再実行回数を追加した。速度は `DECODE_TOKENS_PER_SEC`、round 時間は `DFLASH2_ROUND_MS / DFLASH2_ROUNDS` である。受理 draft 数だけでなく各 round の補正または bonus token も実際の生成速度に寄与する。下表の速度は３回の値、round 時間は平均である。

| prompt | 経路 | 受理 draft/round | ms/round | decode tok/s（３回） |
| --- | --- | ---: | ---: | --- |
| shell | chain | 3.048 | 54.865 | 73.00, 74.25, 74.00 |
| shell | B=0 | 3.048 | 55.126 | 73.53, 73.39, 73.26 |
| shell | B=1 | 3.474 | 62.472 | 71.78, 71.55, 71.42 |
| code | chain | 3.636 | 54.575 | 84.98, 84.84, 84.93 |
| code | B=0 | 3.636 | 55.053 | 84.24, 84.17, 84.14 |
| code | B=1 | 3.554 | 58.709 | 76.82, 77.82, 77.98 |
| reasoning | chain | 3.322 | 55.039 | 78.41, 78.62, 78.46 |
| reasoning | B=0 | 3.322 | 55.476 | 77.91, 77.87, 77.84 |
| reasoning | B=1 | 3.322 | 56.821 | 76.19, 75.99, 75.91 |

shell prompt の B=0 と B=1 の内訳は次のとおり。３回の平均、単位 ms/round。verify GPU 時間には recurrent history の書き込みと attention 等が含まれ、それらは個別に分離できていない。host の待機と GPU 時間は重複し、各行は単純に加算できない。snapshot と commit は非同期処理の投入時間であり、実際の GPU コピー時間そのものではない。

| 計時項目 | B=0 | B=1 | 差 |
| --- | ---: | ---: | ---: |
| round 全体 | 55.126 | 62.472 | +7.346 |
| verify GPU | 44.712 | 45.327 | +0.615 |
| draft 結果の host 待機・コピー | 9.513 | 9.457 | -0.056 |
| 木の候補選択 | 0.004 | 0.110 | +0.106 |
| accept walk | 0.009 | 0.017 | +0.008 |
| 葉の再実行 | 0.000 | 6.660 | +6.660 |
| 木の確定の host 投入 | 0.072 | 0.058 | -0.014 |

B=1 は 57 round 中 10 回、採用葉を再実行した。B=1 の追加は１行で、B=2 以上の追加は最大２行である。確保された全行 history は B=0 で 627,572,736 byte、B=1 で 706,019,328 byte、差は 78,446,592 byte/行。これは容量であり、書き込みだけの実測時間ではない。再実行は既に葉を実際に採用した場合だけ行う。

この測定は既存の１葉の木に限れば３種類とも chain より遅いことを示す。予算付き best-first 木や他の予算はまだ測っておらず、R9700 上のすべての木に利得がないという結論は出せない。分岐後に複数行を採用する木には、数値一致のため、非連続経路の再実行を複数行へ拡張する必要がある。serve 経路への接続と確率的検証も未実施。

### snapshot コピー省略の試行

木では chain の compact log 用 snapshot を復元せず、選択行の history を復元する。木の round だけ snapshot コピーを省く試作を、shell prompt、B=1 で３回測った。変更前は 62.301, 62.501, 62.613 ms/round、71.78, 71.55, 71.42 tok/s、変更後は 62.416, 62.420, 62.412 ms/round、71.64, 71.64, 71.65 tok/s だった。ばらつきに比べて改善は明確でなく、この変更は採用しない。主な負担である採用葉の再実行には影響しない。

試作版に対する `tests/server/test_compute_dflash2_tree.py` は３ prompt の B=0/1/2/4、chain と非投機 target の 256 token 一致を確認した。試作は性能上の効果が明確でないため取り消した。全回帰は実行していない。

## M3: 葉の再実行を除く試作

木の採用経路について、既存の paged KV・target tap の gather と最終行の GDN
recurrent / conv history 復元で確定し、採用葉の再実行を除く。gather は BF16、
FP8（scale を含む）、Octave の既存の各 KV 形式に対応する。B>0 は実験的な
opt-in のままとし、行配置で累算順序が変わることによる数値差を許す。投機検証の
採否規則は変わらないが、非投機との厳密な有限精度分布の一致を保証するものではない。
B=0 は従来 chain との bit 一致を要求する。

数値ゲートは非投機の greedy run で保存した logits と生成列を使う。この run は
非投機自身の生成 prefix に対する teacher forcing と同じである。B>0 の不一致位置の
基準側 top-1/top-2 差を 0.1 以下とする。ただし、最初の不一致以降は２つの生成列の
prefix が異なるため、後続位置の差は同一文脈での数値差の証拠にはならない。
この閾値の検査だけでは最初の分岐後の品質を説明できない点にも注意する。

初回の限定検証では shell prompt の 256 token が chain、B=0/1/2/4 とも
非投機に完全一致した。code prompt では B=0 は一致したが、B=1 は最初に token
118 で分岐し、その位置の非投機 top-1/top-2 差は 0.020216 だった。次の token
119 の基準側の差は 0.826223 であり、指定された「すべての不一致位置で 0.1 以下」
という数値ゲートは不合格である。後続 token は文脈が既に異なるため、これは token
118 の分岐自体が丸め差である可能性を否定しないが、要求されたゲートを満たした
ことにもならない。reasoning prompt の数値ゲートはこの検査の停止により未実施。

予算付き選択の準備として、親 token と位置を受け取る候補確率 callback を使い、
到達確率の積の対数を優先度にして node を追加する CPU 部品を用意した。anchor を
予算に含め、選択済み node だけを展開するため prefix-closed である。単体テストは
異なる親に条件付けた候補、深い枝と兄弟の優先順、祖先・RoPE 位置、非連続な
複数行 walk、異常な候補を検査する。未接続であり、実際の drafter logits の
全語彙を考慮した正規化、親別 selector 呼び出し、容量・CLI 統合、GPU 実測は残る。

初回の性能観測（各１回、受理 draft 数 / round、ms/round）は、shell の B=0 が
192/63、55.191、B=1 が 198/57、55.887、B=2 が 198/57、56.923、B=4 が
198/57、57.349。code は B=0 が 200/55、55.134、B=1 が 195/60、55.774、
B=2 が 198/57、57.062、B=4 が 198/57、57.054。M2 の再実行ありの shell
B=1 は 62.472 ms/round（３回平均）であった。これらは１回ずつの暫定値であり、
反復測定と decode tok/s の比較に代わるものではない。
