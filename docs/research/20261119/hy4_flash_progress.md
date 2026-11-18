# Hy4-flash の進捗

origin/main の 14a23730 から専用 worktree を作成した。原本の config と量子化側の組立仕様を確認した。本体と head の変換完了待ち。組立は Engram を BF16 のまま保持し、DFly を除外するため、どちらも別途処理する。GPU を使わない作業から着手する。

## 現在の待機状態

11 月 19 日 02:17 JST に DFly の 49 射影の追加量子化が完了し、本体との署名照合が通った。02:20 JST に EP=3 の 2Mi 予約が成功した。生成の照合と throughput を確認中。

| 作業 | 完了を確認する場所 |
|---|---|
| 本体の進捗と CPU 追跡 | worktree の `build-hy4/await-body.log` |
| DFly の追加量子化 | `hy4-flash-pilot/shelf-dfly/Hy4-flash-dfly-4bpw/conversion_report.json` |
| EP=3 受入 | worktree の `build-hy4/await-ep3.log`、`hy4-flash-pilot/shelf-qualification/` の測定記録 |
| 3.5 bpw | 4 bpw 版の検証・高速化後に着手 |

R9700 単体では代表 3 層、長い履歴の境界、DFly の確率的棄却を検査した。3 台を使う数値照合と pp2048・decode の目標達成は EP=3 受入で判定する。

## E1　Engram の保存形式

変換器を追加した。原実装の BF16 スケール・E4M3FN 丸め順序に従い、1 行 96 byte の符号と 2 byte のスケールを safetensors に格納する。7 行のテストでゼロ行と最終端数チャンクを含む復元値が一致した。原本の config 上の層番号は 1 始まりで、tensor 名は 1、11、21、31。原本からの 4 表の変換を CPU で開始した。出力先は hy4-flash-pilot/engram-e4m3fn。本体の progress は確認時点で層 26 まで。量子化スクリプト上の最終 report の位置は rocm-exl3-rdna2/models/Hy4-flash-exl3-4bpw-mul24/conversion_report.json。

## K1　EXL3 線形のオフラインビルド

HIP の混在 ISA ターゲット shelf_hy4_flash を追加。K-major の元 trellis を直接読み、suh → H128 → グループ別 A8 → mul24 内積 → H128 → svh の順で処理する。4 bit 本体と 6 bit head、端数入出力のゼロ詰めに対応。ROCm core-7.14 の clang と既存 libc++ 23 で共有ライブラリを作成し、llvm-objdump で gfx1030 と gfx1201 の両 bundle を確認した。数値照合と性能は実機で測る。

## C1　線形・Octave の CPU 基準

実際の層 1 expert 0 gate の trellis で CPU の A8/W4 出力を生成した。HIP 比較器も同じ行列と乱数入力を使用する。Octave は latent を 256 要素 2 ブロック、回転済み位置成分を 64 要素 1 ブロックに分け、各ブロックに 4 bit 符号と FP16 スケールを保存する。294 B/token/layer。sliding は speculative 検証中も必要な過去行が消えないよう 518 行のリングを使う。疎な層は BF16 の pooled index key を 4 token ごとに保存する。

## C2　実 checkpoint での層出力比較

CPU 基準に 4 stream の iHC、Engram の履歴 hash・正規化・残差、sparse/sliding MLA、gated pooling と top-k、shared/latent MoE を実装。量子化側が保存した同一校正列の先頭 4 token と各層の入力 hidden を使った。4 bit 線形と Octave 込みで、BF16 校正出力に対する相対 L2 は層 0 が 0.073091、Engram と sparse MoE の層 1 が 0.030804、sliding MoE の層 2 が 0.017896。各層独立の結果であり、全層を通した精度は未測定。

## P1　実形状に基づく配置の試算

原本の全 shard header と先頭 3 層の量子化対象から、本体の固定重みを 4,329,650,996 B、DFly の固定重みを 424,624,962 B と試算した。routed expert 1 本は 3,557,376 B。2Mi の KV は 7,403,321,016 B、indexer は 1,610,649,600 B。各 32 GiB と仮定し作業域を 2 / 0.5 / 0.5 GiB 除いた場合、配置は 5184 / 7904 / 7904 本。各層の校正頻度を使い、容量制約の下で各 rank の予想読込時間が偏らないよう配分した。未校正層は均等頻度。帯域 640 / 512 / 512 GB/s は初期仮定。VRAM 割当成功の実測は別に必要。

## L1　本体ローダーと rank 別の実予約

shelf-compute hy4 inspect と reserve を追加。inspect は CPU だけで組立済み index・行列の形状と乗数・全 expert・FP8 Engram を読み、配置表に従った正確な重みサイズを出す。reserve は 3 台の ISA を照合し、重み、作業域、KV、indexer を含む領域を同時に確保して解放する。SHELF_HY4_ONLY=ON で専用 shelf-compute のビルドとヘルプ表示が成功した。本体未完成のため実物の inspect・reserve は保留。DFly が含まれるかどうかを出力で明示する。

原本 4 表の CPU 変換が完了。各表の先頭・末尾、チャンク境界を含む 128 行を読み戻し、すべての復元値が元の行別 FP8 演算に一致した。変換した 4 表は合計 64,680,629,984 byte（ヘッダーを含む）。

## N1　ネイティブの生成経路

shelf-compute hy4 generate から本体 42 層と共有 head を呼べるようにした。Octave の符号化・復号、過去グループの indexer、iHC、ホスト FP8 Engram の必要行取得、rank 別の latent 転送・3 射影の grouped expert・加重和を接続した。DFly は予約した本体 KV と採択済み位置の index を読み、6 行の一時ブロックを処理し、最長 5 候補を本体で検証する。実行形式がリンクできた段階であり、3 GPU の生成・性能は未検証。

DFly の slot 0 融合、各後続 slot の hidden correction、過去 index とブロック内 causal attention の契約は、vLLM の Hy4-flash 対応 PR のコミット af127016aa1e14f1e597b546e14f5fdfbd05db11 と原本の tensor を照合した。[実装](https://github.com/vllm-project/vllm/tree/af127016aa1e14f1e597b546e14f5fdfbd05db11/vllm/models/hy_v4/nvidia)

## D1　DFly の追加量子化の準備

quant_dfly.py は本体の report、校正署名、head_complete を照合してから、校正時の層出力を利用して付属 DFly の各行列の Hessian を捕捉する。slot 0 には本体の最終 norm 後の hidden と次 token を入れ、後続 slot の補正には校正列の直前 token を使用する。各 sparse 層の KV を 4 bit Octave で往復してから DFly 固有の kv_b_proj を適用する。実機未使用の検査では、校正済み層 25 の KV と原本 DFly 層 0 で CPU forward を実施し、6 行の出力が有限で、9 射影すべての Hessian 入力を取得できた。5 層全体の校正と mul24 への変換は本体完了後に実施する。

## C3　連続照合と専用ビルド

校正列の先頭 4 token を、途中で BF16 校正状態に戻さず層 0 から層 28 まで CPU で連続実行した。層 28 の BF16 校正 hidden に対する相対 L2 は 0.108770。次回は保存した同一配置・同一入力の続きから検査できる。専用構成の通常ビルドと CTest が成功し、ゼロ行・チャンク境界、sliding リング折り返しをまたぐ 6 行の棄却、2048 境界の選択、容量制約下の expert 配置を検査した。

本体 head の出力・サンプリングを FP32 にした。温度 0 は greedy、正の温度は位置・語彙 ID・seed を用いる Gumbel サンプリングを使う。DFly の提案と target の乱数を分け、採択判定は同じ位置の target サンプルと比較する。2Mi の実予約前に実空き量を調べて枠を超える分だけ expert を再配置し、容量不足時だけ 1Mi で再試行する。

## Q1　実機受入手順

loader で生の norm、router、Engram、indexer、DFly の形状と型を照合するようにした。生成コマンドで最終 prefill 行の FP32 logits を保存できる。同じ重みを保持したまま複数の独立リクエストを反復する機能を、ウォームアップと計測に使用する。lock と load average を検査する locked.py と、CPU logits・DFly の有無・2Mi/1Mi 予約・pp2048・decode の結果を保存する qualify.py を追加。実機受入は本体・DFly 完成後に行う。

## K2　比較用の最適化候補

R9700 向けには既存 WMMA group_body の K-major 入力を有効にし、V620 向けには mul24 の 4 符号復号と整数 dot4 の候補を追加した。専用ビルド、両 ISA のコンパイルに成功。比較用の SHELF_HY4_WMMA=1 と SHELF_HY4_DOT4=1 はそれぞれ独立で、速度・正しさの確認前は既定にしない。

MLA の kv_b_proj について、出力 H128 と svh を整数格子に適用した FP16 行列を使い、nope の query と value の計算順序を入れ替える実験を行った。層 0・2 の実重みと固定乱数 64 行で、直接展開との logit 相対 L2 は 0.001759・0.001658、attention 出力は 0.002236・0.001909。CPU での数値比較であり性能は未測定。長文脈で過去行全体の kv_b_proj を繰り返さずに済む候補として検証する。

## K3　MLA の展開コストを減らす候補

SHELF_HY4_ABSORB=1 で、KV 射影の出力側 H128 とスケールを読み込み時に畳み込み、FP16 の補助行列を rank 0 に構築する経路を追加した。過去の KV は Octave のまま保持する。選択されたスロットの latent を A8 化し、512 次元でスコアと value の加重和を計算してから出力へ展開する。DFly では対応する draft 層の射影を使う。補助行列のバイト数は予約・再配置時の固定費に含める。専用 shelf-compute のビルドと CPU テストを確認した。採用は実機での logits と時間の比較後に判断する。

## K4　R9700 単体の実数値照合

V620 の量子化中に R9700 の lock を取得できた。load average の 3 指標が 8 未満であることを確認し、R9700 の UUID だけを公開した。層 1 expert 0 の実 gate、入力 17 行の CPU 基準に対する出力相対 L2 は、通常の HIP 経路が 0.000001927、K-major WMMA が 0.000002726。コンテナ内で libc++abi の相対リンクが解決しない最初の試行は、toolchain の usr/lib 全体を読み取り専用で bind して解消した。V620 の確認・3 GPU の統合測定は未実施。

DFly 変換を段階別に再開できるようにした。各段階の shard と次の校正 hidden の SHA-256、本体・原本・校正条件を照合し、完了した段階の再量子化を省く。

## K5　R9700 での候補の単体比較

同じ lock と UUID 制限で、実重みの出力を CPU と比較した後、ウォームアップ済みの HIP event を各 7 回測定した。層 2 の kv_b_proj、64 行では通常経路 2.8515 ms、WMMA 0.1155 ms、WMMA の CPU 相対 L2 は 0.0000346。出力 576 の kv_a_proj、5 行も WMMA 相対 L2 0.0000122。層 1 expert 0 の grouped 射影、6 行では通常 0.0691 ms、dot4 0.0337 ms、いずれも相対 L2 0.0000134。これらは R9700 の単体測定で、V620 や pp2048 の達成値には用いない。

EP 転送は両方向の peer access を確認し、利用できない経路では 128 行分の固定 pinned host buffer を経由する。起動時に実際の転送方式を記録し、所有する peer access と buffer を shutdown で解放する。

## N2　R9700 での層単位検査

本番と同じ decoder_layer をテストから呼び、層 0・1・2 の実重みを使って CPU の量子化済み参照と比較した。R9700 1 台に全 expert を置くこの試験は rank 間の転送を含まない。4 行一括と逐次 1 行で同じ相対 L2 を得た。attention と SwiGLU、norm、expert 和の BF16 丸めを参照に合わせた結果、層 0 の相対 L2 は 0.019362 から 0.002729、層 2 は 0.005820 から 0.001509 に減少した。吸収形 MLA を含む候補の層 0 は 0.011446。全層での生成品質は別に確認する。

正の温度での DFly は、各候補を target/draft の確率比で採択し、最初に棄却した位置では両分布の正の差から補正 token を選ぶように更新した。全採択時は本体から追加の 1 token を取る。温度 0 の生成列は通常生成と完全一致を検査し、正の温度では同じ seed の DFly 実行間で再現性を検査する。

## EP1　選択・転送する routing 情報を削減

512 expert の選択と正規化を GPU で実行し、ホストには 16 件だけ返す。各 expert の射影ポインターは読み込み時に rank ごとの表へ格納し、実行時には expert ID を送る。モデル常駐時に実機の ISA と PCI bus ID を調べ、HIP が列挙した順序に依存せず R9700 を rank 0 に対応させる。

R9700 に全 expert を置く層単体検査で、変更前後とも同じ CPU 相対 L2 を得た。4 行の層 1 の中央値は 2.381 → 2.268 ms、層 2 は 2.071 → 1.959 ms。単一 GPU・短い列の参考値であり、PCIe を含む 3 GPU の値は別途計測する。

## A1　長い履歴の境界

実層の重みと固定乱数で作った Octave 履歴を組み合わせ、sparse 層は位置 2047、sliding 層は位置 517 から 6 行を投入した。R9700 と CPU 基準の相対 L2 は、直接展開の sparse 0.007044、sliding 0.002603、吸収形を含む候補では 0.010285 と 0.003998。6 行一括と逐次 1 行の両方で、選択位置の一意性、末尾の絶対位置、長さ上限と出力を確認した。2Mi はこの試験の対象外であり、実モデルを載せた予約試験で別途検査する。

## EP2　同じ expert を選んだ行の再利用

各 rank の選択行を expert ごとに並べ、R9700 では最大 16 行の WMMA、V620 では最大 4 行の dot4 にまとめる候補を追加した。加重和は元の token・expert 順に戻す。R9700 の実 64 行で、SHELF_HY4_GROUP_ROWS=1 の有無を比較。層 1 は 8.753 → 5.803 ms、層 2 は 9.147 → 6.020 ms、CPU 相対 L2 はそれぞれ 0.007659・0.004854。1 行の経路は同じ。V620 の 4 行版と実 EP=3 の性能は未測定。

## A2　疎な indexer と検証行の重複読込

QSA にあった GPU radix 選択を共通化し、Hy4 の上位 511 完了グループを GPU で選ぶ。ホストに全グループのスコアを戻す処理を除いた。検証する複数行の選択スロットの和集合を一度復号し、連続範囲では各 query の位置を加算で求める。R9700 の位置 2047 からの 6 行検証で、sparse 層は 5.719 → 3.135 ms、CPU 相対 L2 は 0.010355。sliding 層は 2.696 ms。共通化に伴い、元の QSA と mul24 projection の translation unit も通常構成でコンパイルした。

## G1　実機受入の待機条件

校正条件・実行形式・配置表・入力の SHA-256 と有効な候補設定を検証記録に保存する。DFly の受理長 0〜5 の頻度と実際に生成した token 数を別々に出す。短い品質検査の入力と pp2048 の入力は別ファイルに残す。

11 月 19 日 00:23 JST 時点で本体は層 31 まで完成し、層 32 の量子化が進行中。CPU の 4 token 連続照合は層 31 で相対 L2 0.137652。ローカルの待機処理が進んだ層だけを追跡し、head_complete と最終 report の両方が揃った後に DFly 追加量子化に進む。受入用には全層の校正頻度で配置表を作り直し、その配置で CPU logits を生成する。DFly 出力が揃ってから EP=3 を実行する。

## D2　確率的検証と追加量子化の実行環境

DFly の target/draft 分布から受理または補正 token を選ぶ本番関数を R9700 で検査した。一致する分布と支持集合が重ならない分布を各 slot に置き、受理長 0〜5 の全ケースで確定列と bonus が一致した。実際のドラフト重みによる受理長は変換後に測る。量子化用コンテナは GPU デバイスを渡さずに起動し、ホストと同じ UID で quant_dfly と乗数 0xDCD12D の量子化モジュールを import できた。

実予約は要求量に加えて rank 別の committed と確保後の空き容量を出す。各生成では rank 別の expert 選択件数と PCIe 経由の latent・部分和の byte 数を prefill/decode に分けて記録する。

## B1　fork の既定構成

専用構成を既定にし、既存のテキスト・PSKLDTOK 入力読込を共有した。gfx1030 と gfx1201 を明示した新規ビルドでも shelf-compute がリンクできた。テキストとバイナリから同じ 3 ID を読み、CPU テストも通過。legacy 構成は SHELF_HY4_ONLY=OFF で選択できる。WMMA を使わない grouped 4 行版は R9700 上でも実層 2 の 64 行で CPU 相対 L2 0.004853 を確認した。V620 実機での成否は別途判定する。

## EP3　校正列上の負荷と PCIe 転送量

層 1〜32 の校正時 top-16 から 32 token ごとに抽出した計 131,072 層・token で、初期配置の分布を確認した。同じ rank 別本数を無作為に割り当てた場合の平均選択数は R9700 / V620 / V620 が 3.87 / 6.06 / 6.07。校正頻度による配置では 5.14 / 5.43 / 5.43 だった。各 token の最大「選択数 / 仮定帯域」の平均は無作為配置の 0.928 倍である。これは配置の指標であり実行時間ではない。

両 V620 が参加する頻度は個別に 99.89% 前後。decode で各参加 rank に 1 本の BF16 latent を送り、部分和を 1 本返すと、両 V620 合計は平均 12,274 B/層・token。41 層に適用した外挿は 40 tok/s で約 20 MB/s。実機では転送量とともに、多数の小さい転送の待ち時間と kernel の重なりを見る。00:36 JST に本体の層 33 が完了し、CPU の連続照合も相対 L2 0.139509 まで進んだ。

## EP4　実測に基づく配置調整の準備

SHELF_HY4_PROFILE_EP=1 のときだけ、各 rank で routed expert の gather から加重和までを HIP event で計る。prefill と decode の GPU 時間を選択件数と併記し、3 台の待ち時間の偏りを診断する。通常のスループット測定では同期を追加しない。placement.py では短い推論用の予約長も指定できるため、2Mi 予約時と pp2048 実行時の配置をそれぞれ検討できる。専用ビルドと CPU テストを再確認した。

## 毎時レビュー第 2 回への対応

1. SHELF_HY4_WMMA、SHELF_HY4_DOT4、SHELF_HY4_ABSORB、SHELF_HY4_GROUP_ROWS は実機比較のための一時スイッチとする。EP=3 で重み・入力・配置・chunk・seed を固定し、各候補の有無を変えて CPU 基準との logits 誤差、greedy 列、pp2048・decode の反復時間を比較する。GROUP_ROWS は WMMA または dot4 と組み合わせて使うため、採用予定の組合せから外した場合も調べる。各候補の採否、適用する ISA・行数、測定記録の場所をこの文書に残す。不採用の専用 kernel、分岐、フラグ、専用の試験設定を削除する。採用した経路は適用条件を通常の dispatch に組み込み、比較専用フラグを外して再度ビルド・受入検査を行う。
2. SHELF_HY4_ONLY=OFF は受入中の共通部品の比較ビルドに使っている。EP=3 受入後に継続利用する呼び出し元と検査を確認し、今後の用途がなければ legacy 構成の入口を整理する。現時点では検証対象を同時に変えないため削除を保留した。

## 毎時レビュー第 3 回

第 2 回の 2 件は、実機比較後の候補削除方針と legacy 構成を保留する理由が確認され、解消となった。01:23 JST 時点で量子化側の last_layer は 37（0 始まり）、head_complete と最終 report は未生成。本体の両完了条件を確認してから DFly を追加量子化する。DFly 完成後、EP=3 の実予約、生成照合、pp2048・decode 測定の順で受け入れる。

## H1　head 完成時の比較条件

最終層と head の完成後、校正時の最終 BF16 hidden に原本の iHC head・norm・FP32 に展開した lm_head を適用する。4 token それぞれについて連続実行した 4 bit 本体・6 bit head との top-1、KL、logits 相対 L2 を CPU で記録する。native 対 CPU の実装差と、この原本対量子化の差を別々に判定する。

## H2　6 bit head の実重み

01:55 JST に head の 6 bit 行列が保存された。SHA-256 と marker を照合し、組立前の実重みで H1 を実施した。先頭 4 token の logits 相対 L2 は原本比 0.236716。BF16 から量子化版への KL は順に 0.02584、1.57486、0.12229、0.26750 で、2 番目の位置の分布差が大きい。top-1 は両者とも 220、2122、6357、12286 だった。これは短い 1 列の結果であり、一般の生成品質の判定には用いない。最終 report と head_complete を確認するまで DFly 量子化は待機する。

## G2　本体組立と DFly 開始

本体は 1161 shard。decoder の符号は 4.0 bpw、量子化線形の実保存量は 4.020704 bpw、未量子化線形を含む値は 4.022769 bpw。head は 6 bit、expert の Hessian fallback は変換 report 上で 17 件。BF16 Engram を含む元の組立済み tensor は 205,729,506,824 B。native loader は必要行だけを読む別の FP8 表を使い、BF16 Engram は常駐させない。最終配置による本体の rank 別の重みは 22,771,088,864 / 28,117,499,904 / 28,117,499,904 B（DFly・作業域・KV を除く）。02:14 JST に DFly の slot 0 融合射影の校正・量子化完了を確認した。

## M1　DFly 完了と 2Mi 実予約

DFly は 49 射影を量子化し、sidecar は index・report 込みで 424,485,632 B。最終配置で CPU の連続照合も再実行した。原本比 logits 相対 L2 は 0.236051、top-1 は 4 位置中 3 位置で一致し、最終位置はともに 12286。H2 の初期配置との違いは rank 別の BF16 加重和を含む。

受入の最初の起動では校正ファイルの読み取り権限が足りなかった。層 0 の校正 capture が保存した同一 input_ids を使い、実際の 2048 ID の SHA-256 を記録するよう修正した。次の試行では共通 GpuArena の gfx1201 制約が rank 1 の確保を拒否した。Hy4 worker は scoped device の確保・解放まで gfx1030 を明示し、既存呼び出しの既定は gfx1201 に保った。

3 台の lock を保持した再試行で、最終配置から 382 expert を再配置し、R9700 / V620 / V620 の本数は 4802 / 8095 / 8095。2,097,152 token の確保成功時の committed は 34,129,931,008 / 29,335,341,056 / 29,335,341,056 B。確保直後の空きは 2,097,152 / 4,972,347,392 / 4,972,347,392 B。DFly と吸収形 MLA の補助行列も含む。2Mi での推論は行っていない。記録は `hy4-flash-pilot/shelf-qualification/ep3-lywnol6v/reserve.log`。

## 毎時レビュー第 4 回への対応

1. 9b0d483a の共通部変更後に、別の build-hy4-default を SHELF_HY4_ONLY=OFF、CMAKE_HIP_ARCHITECTURES=gfx1201 で再構成し、shelf_gpu と既存 shelf-compute の main.hip・compute_runtime.hip を比較ビルドした。6 ステップすべて成功した。既存の引数を省略する ScopedDevice / GpuArena 呼び出しは既定の DeviceArch::gfx1201 を使用し、Hy4 worker の確保だけが gfx1030 を明示する。これは legacy のコンパイル確認であり、既存モデルの生成試験とは区別する。記録は `build-hy4/legacy-arena-config.log` と `build-hy4/legacy-arena-build.log`。
2. EP=3 の同一入力・温度 0 の target-only と DFly の token 列を全件比較する。最終位置の top-1 だけではこの受入条件を通さない。現在は通常生成用の実重みを 3 台へ転送している。

## N3　EP=3 転送の順序を修正し greedy 列を照合

最初の 3 GPU 実行では全語彙の logit が 0 になった。各層の出力を採取すると層 2 から非有限値を含んだ。peer コピー前に送信元の default stream を同期し、宛先 device でコピーを発行するようにした。同じ重み・入力で全 42 層が有限になり、実行時の次 token は CPU と同じ 12286。V620 単体でも実 expert の 6 行 grouped 射影が CPU と相対 L2 0.0000134 で一致した。

修正後の EP=3 では温度 0 の通常生成と実 DFly の生成列が 24 token すべて一致した。4 token 入力の最終 logits は吸収形で CPU 基準に対する相対 L2 0.136028、直接展開で 0.134239。いずれも上位 5 ID は同一だった。相対 L2 は事前の受入目安 0.1 を超えているため、各層に同じ hidden を戻す照合も進めている。記録は `shelf-qualification/ep3-sync-check/`。pp2048 と decode の時間は生成列一致後に独立の計測を開始した。

多数の小さい H2D コピーを避けるため、shard 内の保存順に 3 rank の重みを読み、rank ごとの 16 MiB pinned buffer からまとめて転送する。ウォームキャッシュで測定した重み転送時間は約 20 秒。転送順を変えた場合にも、送信元 stream の同期を外すと非有限値を再現した。調査用の hidden ダンプは本番経路から撤去した。

## P2　最初の反復計測と律速

`ep3-sync-timing` の pp2048 / DFly decode（生成 128、先頭を除く 127 token を計時）の中央値は **103.275 / 16.368 tok/s**。ウォームアップ後の 3 回はいずれも目標未達。プロファイル専用の別実行では、prefill 約 19.8 秒のうち attention 約 9.51 秒、FFN 約 10.28 秒。V620 の expert 演算はそれぞれ約 8.7 秒、R9700 は約 0.50 秒。転送 API の待ちには送信元演算の終了待ちが含まれる。

各層へ同じ入力 hidden を与えた 42 層独立の EP=3 対 CPU 照合（直接展開）では相対 L2 の最大 0.006759、中央値 0.002900。全層を連続して通すと約 0.13 まで増える。独立に、CPU で A8 の各 128 次元の整数内積後にスケールを掛けた場合と、先にスケールを掛ける参照との最終 logits 相対 L2 は 0.142192 だった。元の 0.1 の判定は未達として記録する。これは通常生成と DFly の同一性の検査とは別である。

## P3　実測に合わせた prefill の改善

連続・sliding の prefill は R9700 で直接展開した KV を 16 行ずつ WMMA に掛け、sink を含む softmax と value 積を計算する。chunk 128 の attention 時間は約 9.51 秒から 1.77 秒に短縮した。R9700 と V620 の実効速度比を重く見て、短文脈用の配置表を作成。校正先頭 2048 token で予測した R9700 の選択比率は 34.2% から 62.4% へ増加した。chunk 512 とこの配置で、V620 の同一 expert 最大 16 行を LDS の int8 タイルで処理する。計測同期を含む試行 `ep3-lds16` の 2 回目は pp2048 が 4.615 秒（443.7 tok/s）、47 token の decode が 1.891 秒（24.9 tok/s）。これは初回を除く 1 回の参考値で、最終受入の中央値ではない。decode の吸収形 MLA の score・value 内積も head を 16 本ずつまとめて WMMA に掛け、別途比較中。
