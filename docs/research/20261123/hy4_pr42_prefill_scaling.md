# PR #42 の prefill スケール調査

対象は `3303d2d3`、chunk 1024。実機値は PR #42 の報告、単一 R9700 の microbenchmark は [Phase 2 の記録](hy4_issue40_phase2.md)を引用した。両者の checkpoint 配置、入力、計測条件は完全には照合できていない。物理 4 GPU の critical path は本機では測れない。

## 単一 R9700 の kernel trace

ROCm 10.2、Release、実 4 bpw checkpoint と FP8 Engram。`test_hy4_graph BODY prefillbench LENGTH 1024 1 LOGITS` を使用し、全 42 層の routed expert だけを第 1 MoE 層の重みで共有した。入力 ID は `100 + (37 * i) % 128`。指定 UUID と `/tmp/r9700.lock` を全 GPU 実行に使用した。初回の通常 8k は 18.304 秒。ROCm 7.14 の profiler では kernel trace が取得できず、その実行は 10.685 秒だった。ROCm 10.2 profiler 付きでは 10.998 秒だった。初回の遅延の原因は未確認である。

次は ROCm 10.2 の `rocprofv3 --kernel-trace` で取得した GPU kernel 実行時間の合計である。最初の embedding gather から分類し、重み upload と KV fold は除いた。`hc_mix`、`hc_expand`、`decode_heads`、`router`、`pack_prefill`、`sum_prefill` の順序で範囲を区切り、投入 ID 順に並べ直して範囲内の名前から分類した。kernel の合計であり CPU・通信・待機時間ではない。割合は profiler 付き prefill 壁時計時間に対する比。DSA 用射影は attention その他に、routed down 射影は FFN 前処理に含む。共有 FFN には routed up 射影と最終加算も含む。最終 head などもわずかに混入する。各項目は GPU の仕事量の比較であり物理 4 GPU の時間には置き換えられない。

| 区分 | 2k 秒（%） | 8k 秒（%） | 32k 秒（%） | 128k 秒（%） |
|---|---:|---:|---:|---:|
| absorbed MLA の score・value・query/value 射影等 | 0.629（29.5） | 3.094（28.1） | 14.570（25.0） | 64.639（16.9） |
| attention その他（射影・KV unpack 等） | 0.493（23.2） | 2.531（23.0） | 12.174（20.9） | 55.009（14.4） |
| sparse ID・offset 作成 | 0 | 0.589（5.4） | 10.758（18.4） | 165.562（43.2） |
| DSA score・境界再計算・top-k | 0 | 0.484（4.4） | 3.284（5.6） | 26.535（6.9） |
| routed expert・pack・reduce | 0.493（23.1） | 2.036（18.5） | 7.825（13.4） | 31.453（8.2） |
| router と選択周辺 | 0.293（13.8） | 1.153（10.5） | 4.585（7.9） | 18.448（4.8） |
| 共有 FFN・routed up と合算 | 0.082（3.8） | 0.325（3.0） | 1.296（2.2） | 5.210（1.4） |
| FFN 前処理 | 0.058（2.7） | 0.231（2.1） | 0.911（1.6） | 3.651（1.0） |
| Engram の GPU kernel | 0.013（0.6） | 0.054（0.5） | 0.215（0.4） | 0.855（0.2） |
| その他 kernel | 0.000（0.0） | 0.001（0.0） | 0.004（0.0） | 0.016（0.0） |
| profiler 付き prefill 壁時計 | 2.130 | 10.998 | 58.366 | 383.335 |

Engram の表は H2D・CPU gather・同期待ちを含まず、Engram 全体の時間ではない。8k の prefill 中の全 H2D copy 32 回の GPU copy 時間は計 0.00381 秒で、CPU 側の待ちは含まない。8k の `recall_scores` と `recall_values` は 0.770 / 1.165 秒、32k は 3.404 / 5.537 秒。`sparse_group_ids` 単体は 8k 0.552 秒、32k 10.488 秒、128k 164.178 秒である。128k の該当 kernel は 95,256 回起動され、各呼び出しで履歴幅を単一スレッドが順に scan していた。これは単一 GPU の 128k で最大のコストであり、長さとともに増える主因の一つである。ただしこの結果だけで物理 4 GPU の 239.707 秒の増加分すべてを特定できない。

8k の routed expert の 3 射影 kernel は 1.659 秒で、単純な完全 4 分割なら削減上限は約 1.244 秒（単一 GPU profiler 付き wall の 11.3%）という参考値になる。他の処理や実ルーティング、通信を考慮していない。rank 0 集中部分が大きいため加速余地が限られるという説明と整合するが、物理 4 GPU の実測内訳や 6% の厳密な原因ではない。

## sparse ID 作成の試作

`mask` の履歴幅を 256 スレッドに連続区間として分割し、区間ごとの個数を block scan で prefix sum する。選択された group の昇順と index を従来どおりに維持する。profiler 付きでは 8k の対象 kernel が 0.552 → 0.080 秒、32k が 10.488 → 1.402 秒。128k の対象 kernel は 164.178 → 21.514 秒。profiler 付き全体は 8k 10.998 → 10.512 秒、32k 58.366 → 49.664 秒、128k 383.335 → 240.828 秒（37.18% 短縮）。3 長とも最後の全 120,832 logits が基準版とバイト一致した。128k の比較は profiler 付き各 1 回であり、AB/BA ではない。これは最終 logits の比較であり、全中間行や全条件の正しさを証明しない。profiler を外した 32k の AB/BA は基準 56.505 / 56.513 秒、試作 47.820 / 47.784 秒で、両者の中央値は 56.509 → 47.802 秒（15.41% 短縮）。4 回とも最後の全 logits が基準と一致した。8k では試作 chunk 512 の最後の全 logits も基準 chunk 1024 と一致した。また 3,072 token、chunk 1024 の各層全行 hidden state の 64-bit ハッシュは 42 層すべて一致し、最後の全 logits も一致した。論理 EP=4 の 8k では最後の全 logits と各 rank の選択数が一致した。Graph の既存回帰は論理 EP=1 / 3 / 4 で Target・DFly2 とも 7/7 が通り、capture 失敗時の fallback も通った。 仮に物理 4 GPU の同じ kernel でも単一 GPU と同じ時間差がそのまま critical path から除けるなら、報告値は 8k で約 10.09 → 9.62 秒、32k で約 54.17 → 45.09 秒、128k で約 365.85 → 223.19 秒という算術例になる。実機の入力や重み配置、競合、同期が異なるため予測の保証はなく、物理 4 GPU で再測定する。

## 論理 EP=4

同じ R9700 を 4 論理 rank として使う基準版 8k は profiler 付きで 14.166 秒。kernel 時間の和は 15.357 秒で、stream 間で重なるため wall より長い。expert の 3 射影 kernel の合計は 4.117 秒で EP=1 の 1.659 秒より大きい。4 論理 rank が 1 枚の演算資源を共有し、各 rank の固定作業も増えるためであり、物理 4 GPU の並列速度や PCIe の推定には使えない。rank 0 の queue の kernel 合計は 11.469 秒、peer 3 queue は順に 1.318 / 1.364 / 1.207 秒。

rank 0 queue を同じ名前と範囲で分類すると、MLA 3.759 秒、attention その他 2.854 秒、sparse ID 0.771 秒、DSA 0.627 秒、rank 0 MoE 1.417 秒、router 周辺 1.252 秒、共有 FFN・合算 0.466 秒、FFN 前処理 0.264 秒、Engram kernel 0.058 秒、その他 0.001 秒だった。rank 0 の送受信や結果加算は範囲分類内に含まれ、通信待ちを独立に切り出していない。これらは重なる queue の GPU 時間であり、14.166 秒の wall に対して単純加算できない。

## 既存の全体時間と制約

| 入力 token | 物理 4 GPU（秒） | 2k 時間を比例換算（秒） | 差（秒） |
|---:|---:|---:|---:|
| 2,048 | 1.971 | 1.971 | 0 |
| 8,192 | 10.088 | 7.884 | 2.204 |
| 32,768 | 54.174 | 31.536 | 22.638 |
| 131,072 | 365.851 | 126.144 | 239.707 |

差には固定費や入力の違いも含まれ、attention の実測時間ではない。位置の増加に伴う追加処理が大きいことを示す比較に限る。8k の 1 GPU 10.736 秒から 4 GPU 10.088 秒への差は 0.648 秒、6.04% で、3 GPU から 4 GPU は 0.161 秒、1.57% である。異なる環境・配置のため、この差をそのまま並列化可能部分とは見なせない。仮に同条件で通信費ゼロ、完全均等分配という Amdahl モデルを当てはめた場合に限り、見かけの並列部分は単一 GPU 時間の約 8.05%、0.864 秒となる。これは物理的な内訳の推定値ではない。

Phase 2 の同一 GPU 上の 1024 行 MoE microbenchmark では、MoE 全体 9.935 ms、routed expert の 3 射影カーネル 4.181 ms、共有 FFN 単独 0.678 ms。対象は第 1 MoE 層の実 router と実重みであり、全層・全入力の代表値や物理 4 GPU の壁時計時間ではない。特に射影カーネルの和は入力回転、経路計画、通信、同時実行を含まない。共有 FFN の値を 41 層・各 chunk に単純に掛けると 2k / 8k / 32k / 128k でそれぞれ 0.056 / 0.222 / 0.890 / 3.558 秒である。これは重ね合わせによる節約可能量の予測ではない。

## 転送量の算術モデル

Phase 2 の論理 EP=4、1024 行、第 1 MoE 層では push の論理 payload が 18,860,568 byte と報告されている。各 chunk の 41 MoE 層で同じ payload と仮定し、報告された 26 GiB/s を全 payload に直列適用すると次の値になる。これは大きな 128 MiB 転送で得た帯域を小さな複数転送に当てはめた理想的な帯域換算であり、PCIe 実測時間、下限、通信待ち時間のいずれでもない。実 routing、重複送信、peer ごとの同時転送、共有リンク、転送粒度と同期を実機で測る必要がある。

| token | 仮定した全 payload（GiB） | 26 GiB/s 換算（秒） | 全体時間に対する換算比 |
|---:|---:|---:|---:|
| 2,048 | 1.440 | 0.055 | 2.81% |
| 8,192 | 5.761 | 0.222 | 2.20% |
| 32,768 | 23.046 | 0.886 | 1.64% |
| 131,072 | 92.183 | 3.545 | 0.97% |

この換算だけで通信は無視できるとは判断できない。転送待ちには小粒度アクセス、投入と event、共有帯域、末尾 peer 待ちが含まれ、容量帯域とは別に測る必要がある。

## コード上の依存関係

- rank 0 は attention、DSA、Engram、共有 FFN、router、入力射影、経路 pack と結果加算を実行する。`prefill_moe` は各 peer の転送と expert を順にホストから投入し、rank 0 の expert を投入後、peer の ready event を順に待って加算する。peer ごとの expert は独立 stream に投入されるので、ホストの for 文だけを根拠に GPU 計算が完全直列とは言えない。push の送信は rank 0 stream 上に並び、各 peer の受信・実行は event に依存する。実際の重なりと最遅 rank は trace が必要。
- prefill の共有 FFN は全 routed expert と peer 結果加算を同じ rank 0 stream に投入した後に投入される。decode の専用 stream と違い、共有 FFN と routed expert を明示的に重ねていない。計測用 `SHELF_HY4_PROFILE_EP` は各層や expert の同期を追加するため、その時間を通常運用の wall time と直接比較できない。また `transfer_ms` は転送関数のホスト投入時間であり GPU 転送完了までの時間ではない。
- Engram は 4 層で、CPU 上の gather を既定で非同期に先読みするが、最初の `take_engram` は全 4 層分の future を待つ。各層の `Ops::copy` は H2D 投入後に rank 0 stream を同期する。1024 行あたり 4 × 1024 × 16 × 98 = 6,422,528 byte、8k 全体で約 49 MiB。26 GiB/s で約 1.84 ms は単なる容量換算で、ページ可能な host メモリの実効速度、CPU gather や stream の先行作業待ちは含まない。これだけでは無視可能とも支配的とも断定できない。
- DSA は位置 2048 以降に sparse 選択を行い、16 行単位で候補全体の score、境界再計算と top-k を実行する。候補数は履歴長に応じて増える。一方 sparse attention は 4096 を越えると 16 行単位に分割され、各行は最大約 2048 token を参照する。sliding 層の窓は 512 token。長文では DSA と sparse ID 作成に加え、attention の unpack や MLA が増える。Phase 2 の DSA 選択 microbenchmark は 16 行につき位置 8192 / 32768 / 131072 で 0.1337 / 0.1873 / 0.4435 ms だった。固定入力・単一位置の値であり全層の積分値ではない。
- `5d516cb4` は旧 tiled WMMA の明示的 KV 射影を廃止し、prefill を absorbed 経路に統一した。現経路にも head 単位の WMMA があり、単純な「WMMA から scalar への変更」ではない。旧方式は KV の BF16 丸め位置が異なり、chunk 境界で不一致が生じるため修正された（[PR #39 の検証](../20261122/hy4_pr39_correctness.md)）。性能回帰の有無は同一入力の比較がないため未確定で、旧方式の無条件復元は正しさを損なう。

## 優先順位と次の計測

1. sparse ID 作成の試作を物理 4 GPU で AB/BA し、長文と chunk 境界における正しさを評価する。128k は profiler 付き 1 組のため、通常実行でも繰り返す。
2. 物理 4 GPU では同じ入力・配置・転送設定の通常実行について各 GPU と host のタイムラインを収集し、通信の重なり、最遅 peer、rank 0 idle、H2D と同期を特定する。計測の同期追加による摂動は通常実行と比較する。全時間を重複なしで分解するには GPU kernel 時間の総和ではなく critical path が必要。
3. 共有 FFN を独立 stream に移す候補は、入力準備完了 event、別 arena、結果合流 event、リソースの寿命を明示し、結果と chunk 境界の正しさを検証する。rank 0 expert や通信との GPU 資源競合により、0.678 ms 全体を節約できるとは限らない。AB/BA と実機で判断する。
4. DSA の境界再計算・top-k、absorbed MLA の 16 行ごとの起動と同じ KV の再読込を profile に基づいて改善する。旧 tiled 経路の比較は実装順序の違いと精度要件を併記し、短・長文で全 logits、chunk 一致、層出力を検査する。Engram の pinned staging と前倒し転送も実測後に検討する。


## 再現と実行状況

`BODY` は 4 bpw 本体と 4 層の FP8 Engram を含むディレクトリ、`DRAFT` は DFly2 のディレクトリ。基準版と試作版はそれぞれ別のバイナリを使用した。以下は試作版の代表例である。

```bash
export ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" prefillbench 32768 1024 1 /tmp/hy4-logits.f32
flock /tmp/r9700.lock /opt/rocm/core-10.2/bin/rocprofv3 --kernel-trace -f csv -d /tmp/hy4-trace -o trace -- build/tests/test_hy4_graph "$BODY" prefillbench 131072 1024 1 /tmp/hy4-logits.f32
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" "$DRAFT"
```

出力末尾の logits 比較では全位置の logits を確認していない。3,072 token の層別ハッシュは衝突可能性がある。入力は反復列のみで、物理 4 GPU の通信・負荷分散、128k の通常実行 AB/BA、旧 tiled MLA の性能差は未検証。試作は未コミットで、物理構成での再測定と追加入力の正しさ確認が必要である。

`test_hy4_reduce` と差分の空白検査も通った。調査用の一時ファイルを削除し、GPU 排他ロック内で KFD プロセス 0 件を確認した。

## PR #39 取り込み後の event と回帰

PR #39 の転送 event 再利用の修正と rank 別 expert 射影表を取り込んだ。prefill の peer 返却でも ready と ack を再記録する前に対象 event の完了を待つ。`hipEventRecord` の直前で `hipEventQuery` を呼ぶ一時的な計測器を使い、20 token、chunk 20、論理 EP=4 で全 event を検査した。push の 369 回、peer の 492 回はいずれも未完了 0 回、照会エラー 0 回だった。両方式の最後の全 logits が一致した。再発確認用に `test_hy4_graph` は `SHELF_HY4_CHECK_EVENT_REUSE=1` のとき、未完了 event の再記録を失敗として扱う。

射影表を rank ごとに分離した後、8,192 / 32,768 token と論理 EP=1 / 4 の全組み合わせで chunk 512 と 1024 の最後の全 logits が一致した。EP=1 の両入力長は変更前の記録とも一致した。Graph の論理 EP=1 / 3 / 4 回帰、peer 転送と reduce、および CPU 必須 CTest 133 件が通った。CPU CTest はビルドに指定された LLVM の共有ライブラリディレクトリを `LD_LIBRARY_PATH` に指定した。上記の性能値は event 同期の修正前の測定であり、修正後の物理 4 GPU 性能は未測定である。
