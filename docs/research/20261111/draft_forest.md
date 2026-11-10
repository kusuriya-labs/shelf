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

M1 は未完了。CPU 上で chain と兄弟葉の親・深さ・祖先物理位置・RoPE 位置を組み立て、target token に一致する子をたどる純粋関数と単体テストを追加した。推論経路には接続していないため、木検証はまだ利用できない。tree の実機での正しさの検証、計測および既定値の判断は未完了である。

## 分岐ごとの候補 score の準備

candidate selector は従来、1 行目の anchor 以降を直前行の選択結果で逐次採点する。分岐先ではその直前行が親とは限らないため、行ごとに GPU 上の親 token を指定して独立に採点できる経路を追加した。各行を別 block で処理し、同じ draft 位置を複数の親で採点できる。既存の batch request 経路とは同時に指定できない。親 token と候補 token の範囲は呼び出し側が保証する。

GPU 単体テストでは同じ候補列について親 3 と親 5 を指定し、それぞれ異なる候補が選ばれ、各候補の score が CPU 参照に一致することを確認した。既存の chain も同じテストで確認した。この API はまだ drafter や verify に接続されておらず、tree 生成・検証は有効にならない。確率値への正規化（top-k 外の残余確率を含む）も実装していない。

残作業は、候補の GPU バッファと prefix-closed な木の構築、行別の祖先・RoPE・GDN history と容量確保、採用経路の KV と tap の gather、GDN の最終行復元、深さに対応する sampling、greedy・確率 sampling の統合テスト、低・高同時数の性能と VRAM 測定である。性能判断に必要な測定がないため、既定値の変更は行わない。

## M1 の未完了項目

候補 top-2 の取得と単一 request への接続、全行 recurrent history の確保、tree metadata の GPU 転送、選択経路の KV と tap の確定、行数上限と memory plan の調整が必要である。既存の `restore_gdn_spec_history` は conv と recurrent の両方を復元できる。統合後に greedy 一致・B=0 の統計一致・受理長と round 時間の測定・全回帰を実施する。M2 では serve 経路、予算付き選択、確率的検証、性能最適化を扱う。

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
