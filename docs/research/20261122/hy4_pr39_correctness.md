# Hy4 の EP 転送と prefill logits の照合

2026 年 11 月 22〜23 日に PR #39 の `93a01b03` を基準とし、ROCm 10.2、gfx1201、`SHELF_HY4_ONLY=OFF`、R9700 1 台で測定した。GPU 実行時は対象 UUID を指定し、`/tmp/r9700.lock` を取得した。3 台・4 台での外部測定条件は PR #39 の実機報告に記載されている。

## 1. EP 転送

`5d516cb4` で受信側 stream の `hipMemcpyPeerAsync` を使う 2 か所を受信側 GPU の読み出し kernel に置き換えた。対象は MoE の転送と TP/EP の加算前転送である。物理 R9700 ×4 ではこの変更後も peer のみ失敗した。[PR #39 の実機報告](https://github.com/kusuriya-labs/shelf/pull/39)

旧実装で peer access が有効でも転送後に routing が破損する実機での切り分けは[以前の調査](../20261120/hy4_rotate_output_investigation.md)にある。単一 GPU 内の論理 rank 試験では PCIe や SDMA の物理障害を再現できない。異なるカード間での修正後の読み出しは実機での追試が必要である。

同じ配置で 3 方式の単一・バッチ生成を比較するには、実機の可視デバイスを指定して次を実行する。`OWNERS` にはその台数に対応する配置表を渡す。Target Graph は対応する push 構成で起動し、peer と host の Target は eager で実行する。

```bash
cmake --build build --target test_hy4_graph
flock /tmp/r9700.lock build/tests/test_hy4_graph "$BODY" "$DFLY" transport "$OWNERS"
```

### EP-1　記録中の event の再使用

**ACCEPT**。eager の転送は、rank の組と方向ごとに 2 組の ready / ack を交互に使う。GPU が最初の組を処理中でも、ホストは 3 回目の転送を投入できた。ローカル expert 用の ready / done は毎層同じ組を使っていた。HIP は記録中の event への再記録を未定義としている。各組の再使用前に、その組で最後に記録した event の完了を待つようにした。転送では peer と host の場合は ack、push の場合は ready が該当する。[ROCm 10.2 の event 仕様](https://rocm.docs.amd.com/projects/HIP/en/docs-10.2.0/reference/hip_runtime_api/modules/event_management.html)

R9700 1 台、論理 EP=4、peer、Graph と DFly 無効、20 token 入力で、各記録の直前に `hipEventQuery` を入れた。表は `hipErrorNotReady` の回数と記録回数。B=1 と B=8 は各スロットで 4 token を生成した。修正後は同じ全条件で未完了の再記録が 0 回だった。

| 修正前の実行 | 転送 ready | 転送 ack | local ready / done 合計 |
|---|---:|---:|---:|
| prefill のみ | 0/246 | 0/246 | 使用なし |
| B=1 | 594/984 | 594/984 | 222/246 |
| B=8 | 594/2706 | 594/2706 | 222/246 |

`logical_ranks` の worker にも個別の射影表を確保し、未所有 expert のエントリをゼロにして、実機と同じ所有条件で照合した。物理 GPU と重み本体は共有している。論理 EP=3・4、3 転送方式、DFly の有無、バッチ数 1・2・4・8・16 の 60 条件で全スロットの生成列が一致した。

### 物理 4 枚での切り分け

PR #39 の通常 prefill は各 MoE 層で rank 0 の picks を D2H して stream を同期し、その後で CPU が経路を検査する。上記の 20 token prefill だけでも記録中の event の再使用は観測されなかった。PR #42 の `gather_rotate_pair` では実行 rank が所有しない expert が route に入ると、そのエントリの gate / up の `suh` が null になる。実機の報告には fault 時の route、命令 PC、両端の転送内容がなく、どの値が最初に壊れたかは未確定である。[PR #42 の実機報告](https://github.com/kusuriya-labs/shelf/pull/42)

PR #42 の `3303d2d3` では通常 prefill の計画も GPU 上で作る。既存の `prefillbench 20 20 4` を peer、論理 EP=4 で実行し、同様に記録直前の状態を調べた。scatter の `Residency::transfer` では ready と ack の各 123 回中 99 回、返却部分和を直接読む `prefill_moe` では ready と ack の各 123 回中 111 回が未完了だった。後者は `Residency::transfer` を通らず、常に添字 0 の event を使う。PR #42 に反映する際には、この専用経路でも前回の event の完了を保証する必要がある。

まず実機で、モデルを載せずに全方向の peer 読み出しを試す。送信側 kernel は同じ領域へ毎回異なる値を書き、受信側も同じ領域を繰り返し使う。試験は 8 回分の受信内容を期待値と照合し、最初の不一致で方向・転送回数・offset を表示する。

```bash
cmake --build build --target test_hy4_peer_transfer
ROCR_VISIBLE_DEVICES=0,1,2,3 flock /tmp/r9700.lock build/tests/test_hy4_peer_transfer
```

全方向で一致して推論だけが失敗した場合は、同じ入力・配置で Graph を無効にした peer と host を比較する。診断用ビルドでは、転送元と転送先それぞれの stream 上で、バッファの再使用前に送信直前・受信直後の内容を保存する。最初に異なる層・方向について、PR #39 では decode の plan と latent、通常 prefill の latent と返却部分和を比較する。PR #42 の prefill はその rank の mask、有効な choice と latent 行だけを比較する。省略された領域の古い内容は期待値の対象外である。転送内容が一致したら、gather の直前で count・row・expert・owner と当該 rank の gate / up の suh を確認する。

## 2. Attention の 16 行境界

**ACCEPT**。旧 tiled 版は各 token の KV 射影を先に計算して BF16 に丸めた。absorbed 版は FP16 に fold した重みを query と集約後の value に掛けた。この積の括り方と丸め位置の差を、実 4 bpw checkpoint の層 0・1・2 の `kv_b_proj` と sink を用いて測定した。入力は固定乱数の BF16 query、latent、RoPE 成分、gate を 20 行用意した。CPU 基準は EXL3 の復号、128 要素単位の Hadamard 変換と入力の INT8 量子化を行い、射影以降の積、causal mask、sink を含む softmax、value 集約、gate を FP32 で計算した。

| 層 | 旧 absorbed 対 CPU FP32、chunk=1 | 旧 tiled 対 CPU FP32、chunk=20 | 旧 chunk=20 対 1 | 修正後 chunk=15・16・20 対 1 |
|---:|---:|---:|---:|---:|
| 0 | 0.003558 | 0.003935 | 0.003459 | すべて 0 |
| 1 | 0.003474 | 0.003911 | 0.003605 | すべて 0 |
| 2 | 0.003630 | 0.003987 | 0.003758 | すべて 0 |

表は相対 L2 誤差。固定した入力では旧両経路とも単層で 0.4% 未満だった。CPU 側で旧明示射影と同じ位置に BF16 丸めを入れると、旧 tiled 版の相対 L2 誤差は順に 0.000043、0.000103、0.000105 になった。修正では decode の射影順序と head をまとめる WMMA カーネルを prefill にも適用した。

Attention を揃えた段階でも、32 token の回帰入力では 16 行を超える線形射影で残差が出た。K 方向の分割数が 16 行以下とそれ以上で異なるためである。同じ行列で分割数を固定し、部分和の加算順序も揃えた。fused GEMM の選択とは独立に加算順序が変わっていた。

sliding 層では別の境界も修正した。位置 511 以降に 8〜15 行をまとめると、旧版の固定 518 行の unpack 領域に対し、各行の窓が参照する KV の和集合が 519〜526 行になる。eager 実行時は和集合の実寸で予約する。この範囲外参照は 20・32 token の先頭 prefill では成立しない。

### 全 42 層と最終 logits

20 token を入力し、層を進めるごとに当該層の全 512 expert の実重みを同じ GPU 領域に載せ替えた。各層でチャンク条件ごとに独立した hidden state を保ち、各条件の先頭から KV を作り直した。全層で同じ expert 重みを共有する Graph 試験とは別の測定である。EP は `expert_id % EP` による分割を使用した。

```text
8045 846 198 814 20139 11 303 883 220 18 15 15 4105 11 1204 63520 45543 23944 685 436
```

| EP=3 | 旧版の最終 logits 対 chunk=1、相対 L2 | 旧版の最大絶対差 | 旧版の greedy ID | 修正後の最大絶対差 | 修正後の greedy ID |
|---|---:|---:|---:|---:|---:|
| chunk=1 | 0 | 0 | 30 | 0 | 30 |
| chunk=15 | 0 | 0 | 30 | 0 | 30 |
| chunk=16 | 0.258562 | 3.25731 | 11 | 0 | 30 |
| chunk=20 | 0.306666 | 4.51822 | 9468 | 0 | 30 |

同一 GPU の論理 EP=3、push、expert 共有の試験で、初回を除く 3 回の prefill 全体の中央値も測定した。128 token は 0.221→0.211 秒、512 token は 0.560→0.617 秒、1024 token は 1.127→1.223 秒だった。同じ射影順序で query 行専用のタイル分岐を残した試作は順に 0.216、0.628、1.237 秒だったため棄却した。長い入力ではチャンク間の数値一致と引き換えに、この条件で約 9〜10% の時間増を観測した。物理 EP の運用性能は別途測定する必要がある。

### checkpoint 付属 tokenizer と配置表での追試

次の 32 token の先頭 16・20・32 token を用いた。checkpoint 付属の `tokenizer.json` で英語の文章を変換し、special token の自動追加を無効にした。3・4 rank の配置表はそれぞれ `placement.py --capacity 32 32 32`、`--capacity 32 32 32 32` に同じ DFly と `--tokens 2097152` を指定して作成した。SHA-256 の先頭 12 桁は tokenizer が `2e3f3d22065c`、3 rank の配置表が `e91ea90e5bf3`、4 rank の配置表が `1c49373b9d3d` である。

```text
802 8778 299 9480 341 39144 13 360 46544 52315 533 10044 565 1823 269 7370 587 269 10575 2417 18221 302 269 18212 13 44635 269 1419 4782 269 46544 2929
```

20 token の実入力から、層 0・1・2 の RoPE 適用後の query、Octave 復号後の latent と RoPE 成分、gate を採取した。同じテンソルを CPU と両 GPU 経路に直接与えた。20 行における修正前の tiled / absorbed の相対 L2 差は順に 0.002170、0.002676、0.002866、16 行では 0.002218、0.002737、0.002904 だった。修正後は 16・20 行とも、3 層すべてで chunk=1 とビット一致した。修正後の 20 行の CPU FP32 基準に対する相対 L2 は、順に 0.003047、0.003360、0.003431 である。

全 42 層の実重みで、32 token の全行を chunk=1・8・15・16・20・32 で照合した。各 EP 構成内では、全 32 行の logits がビット一致した。16・20 token だけを入力した独立の forward も、32 token の対応する先頭部分とビット一致した。

| 入力長 | EP=3 の greedy ID | EP=4 の greedy ID | EP 間の最終 logits の相対 L2（分母は EP=3） | 最大絶対差 |
|---:|---:|---:|---:|---:|
| 16 | 715 | 587 | 0.144423 | 2.75415 |
| 20 | 14321 | 55053 | 0.121735 | 2.24450 |
| 32 | 1823 | 1823 | 0.111520 | 2.12131 |

16 token の同一入力から各スロットで 4 token を生成する検査も行った。最初に、router が選んだ各層固有の実 expert 重みをオンデマンド転送して参照列と利用集合を採取した。次の実行では、この入力に使用する expert を実配置表どおりの独立した論理 rank の領域に常駐させた。固定重みは全て元の checkpoint を使い、実行中に利用集合外の expert を選んだ回数を GPU で検査した。

EP=3・4、push / peer / host、DFly2 有効・無効、B=1・2・4・8・16 の 60 条件すべてで全スロットの生成 ID は参照列と一致した。単一スロットの Graph 設定を別途切り替えた 24 条件も、初回の全 logits と生成列が一致した。peer で Engram 先読みを無効にした追加 20 条件でも全スロットの生成列が一致した。利用集合外の参照は 0 件。生成列は EP=3 が `715 17707 67337 11`、EP=4 が `587 269 10044 6896` だった。これらは同一カード内の転送と stream の検査である。

## 3. 起動ごとに観測された外れ値

**原因を特定**。実機検証者の報告では、DDR5 4800 MT/s の 2DPC・2R 混在、非 ECC のホストメモリ構成で bit 6 の反転が発生し、3600 MT/s では再現しなくなった。切り分け専用の `hy4_checkpoint_checks.cpp` と `test_hy4_graph` の `upload` / `layers` 診断経路は削除した。報告者の 32 token 入力では、absorbed のみの chunk=8 と 15 に同じ外れ値のハッシュが現れた。この現象を tiled 固有の race とする説明を棄却した。

TP=1 の初期化を監査した。重みの登録順は shard と tensor 名から決まり、rank ごとの領域を順番に確保する。アップロード順は配置時に記録した順序で固定されている。16 MiB の pinned buffer からの H2D は同期コピーを使う。expert の配置表・射影表の転送後に各 rank で fold を実行し、`hipDeviceSynchronize` が完了してから推論に入る。Engram の非同期先読みはこの後に開始する。20・32 token の先頭 prefill では sparse top-k 選択の開始位置 2048 に達しない。複数 query タイルは別々の行に書き、末尾タイルの無効な query lane にはゼロの fragment を与える。MoE では 6 行以下と 7 行以上で plan の作成方式が切り替わるため、両側を別々に検査した。

基準コミットを独立に 16 プロセス起動し、32 token を chunk=8・15・16・32 で処理した。論理 EP=4、host 転送、Engram 先読みを交互に有効・無効とした。各プロセスで重み領域と fold の出力先を転送前に 0x7f または 0xff で埋め、各 forward 前には作業域、KV、index を埋めた。この試験は第 1 層の expert を共有する。転送した 7,125 tensor の全 6,575,455,606 バイトがホストと一致し、47 個の fold 結果の結合ハッシュは全起動で同じだった。旧版でも各 chunk の全 logits は起動間で 16/16 一致した。修正後の同条件でも独立した 16 起動、さらに 1 プロセス内 8 反復で各回の全 logits を保存した。chunk=8・15・16・32 のハッシュはすべて一致した。別途、47 射影の全 FP16 要素を CPU で復号・変換して照合した。別の試験では 4 論理 rank それぞれに独立した重み領域・stream を確保し、実装の rank 別アップロードを通した。起動時の読み戻しや poison を挟まずに最初の forward を測定し、host・push と Engram 先読み設定を交互に変えた。基準コミット、修正版とも独立した 16 起動のハッシュは `c7a6751a004c` で一致した。これらの試験だけでは報告者の外れ値の原因を特定できなかった。

実配置表については、1 台に rank 分を順次載せ替え、EP=3・4 とも GPU 常駐対象の全 79,430,516,086 バイトを checkpoint と照合した。各構成で不一致は 0 バイト。配置・射影表および 47 個の fold 結果も CPU 基準と一致した。ホスト常駐の Engram 表はこのバイト数に含まれない。

付属 tokenizer の 32 token では、全 expert を使える層単位の実行を正解とし、実際に選ばれた層固有の expert 重みを論理 rank ごとに常駐させても初回 logits が一致するかを検査した。EP=3 の利用集合は 6,241 個、EP=4 は 6,243 個。独立起動では host / push、Engram 先読みの有無、予約領域の既定値 / 0x7f / 0xff を切り替えた。半数は計測用 GPU カーネルも読み戻しも挟まず最初の forward を実行した。基準コミットは各 EP で独立 16 起動 × chunk=8・15、最終版は各 EP で独立 16 起動 × chunk=1・8・15・16・20・32 を検査した。両版合わせて 64 プロセス、保存した全 256 件の logits は、各 EP の全重みによる参照とビット一致した。SHA-256 の先頭 12 桁は EP=3 が `32e3c8fc16db`、EP=4 が `c0db93be2ff2`。この入力と単一物理カードでは、旧版の段階から外れ値を再現できなかった。

## 4. EP 数と greedy ID

**加算順序の差を確認**。修正後の同じ 20 token、chunk=20 で、層 0 の hidden state は EP=3・4 間でビット一致した。最初の MoE 層で各 expert の出力と routing weight を読み戻し、rank 内の積、加算、rank 間の加算の各段階を BF16 に丸めて CPU で再計算した。3・4 rank の両方で GPU の合算結果 30,720 要素すべてが CPU と一致した。

EP=3 と EP=4 の hidden state の相対 L2 差は層 1 で 0.001911、層 2 で 0.014673 となった。42 層後の最終 logits の相対 L2 は 0.276933、最大絶対差は 3.83231 だった。20 token 目の greedy ID はともに 30 だった。同じ全行の最終 hidden state に head を適用すると、9 token 目では EP=3 が 198、EP=4 が 19 となり、12・17 token 目でも top-1 が変化した。付属 tokenizer の 32 token と実配置表で選択 expert の集合も照合した。最初の MoE 層では 32 行すべて一致し、層 2 では 10 行の top-16 集合が変わった。全 41 MoE 層の延べ 1,312 行では 804 行が異なった。rank 内の BF16 部分和の括り方で生じた初期差が後段の routing まで伝わったことを確認できる。

## 回帰試験

- Release 全体ビルドと CPU 必須テスト 133 件が通った。Hy4 の CPU 単体テストも通った。
- TP=2・EP=2 の 4 論理 rank で peer 転送を使い、65,539 要素の BF16 合算を 6 回繰り返した。全 rank が CPU の期待値と一致した。
- 1・3・4 論理 rank で Target と DFly2 の Graph / eager が各 7/7 一致し、capture 失敗時の eager 復帰も通った。sliding window の位置 511 からの chunk=1・8・15・20 も一致した。
- expert 共有のサーバー試験で JSONL の同時生成・途中キャンセル・待機・スロット再利用・DFly2 が通った。ビルドした HTTP サーバーで `/v1/meta`、4 並列 SSE、途中切断後の非ストリーム生成を確認した。
