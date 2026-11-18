# Hy4-flash の進捗

origin/main の 14a23730 から専用 worktree を作成した。原本の config と量子化側の組立仕様を確認した。本体と head の変換完了待ち。組立は Engram を BF16 のまま保持し、DFly を除外するため、どちらも別途処理する。GPU を使わない作業から着手する。

## 現在の状態

11 月 19 日 03:26 JST に 4 bpw の最終受入を完了した。採用版は 2Mi 予約、pp2048 676.62 tok/s、DFly decode 50.65 tok/s、候補撤去前後と通常生成・DFly の 256 token 全一致を確認済み。03:28 JST から別出力で 3.5 bpw の変換を実行中。

| 作業 | 完了を確認する場所 |
|---|---|
| 本体の進捗と CPU 追跡 | worktree の `build-hy4/await-body.log` |
| DFly の追加量子化 | `hy4-flash-pilot/shelf-dfly/Hy4-flash-dfly-4bpw/conversion_report.json` |
| EP=3 受入 | worktree の `build-hy4/await-ep3.log`、`hy4-flash-pilot/shelf-qualification/` の測定記録 |
| 3.5 bpw | worktree の `build-hy4/quant-mixed.log` と `build-hy4/mixed-cpu-chain.log`。完成した checkpoint と変換報告は `hy4-flash-pilot/shelf-mixed/` |

R9700 単体では代表 3 層、長い履歴の境界、DFly の確率的棄却を検査した。4 bpw の EP=3 での数値照合と速度の結果は G3 に記録した。

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

## P4　速度目標に達した実測と残る受入

512 window をまたぐ長い prefill では必要な最後のリング行だけを保存するようにし、chunk 1024 を実行した。不可視の QK / PV タイルを省略し、6 bit head にも WMMA を適用。rank 間は送信元の system-release event を宛先 stream で待ってから peer copy を発行し、同じ層の routing 情報を一つの pinned packet にまとめた。短い入力の EP=3 通常生成・DFly 列は 24/24 一致した。

小さい行数で出力タイル数が少ない射影を split-K にした `shelf-qualification/ep3-split/bench.log` では、初回を除く 3 回の中央値が **pp2048 676.35 tok/s、DFly decode 50.64 tok/s**。decode は 128 生成のうち prefill で得た先頭を除く 127 token で測った。33 検証ラウンドで 94 候補を受理。直前の event 版は 47 ラウンド・79 候補受理で 676.75 / 27.55 tok/s。decode の差には計算時間の短縮と生成列・受理長の変化の両方が含まれる。固定配置での候補ごとの logits・生成列・時間比較と、採用時の通常 dispatch への組み込みは未完了。

CPU logits 相対 L2 の初期目安 0.1 を超えた場合はその事実を結果 JSON に保存し、独立した greedy 列と throughput の検査も最後まで実施する。閾値を超えた実行をこの判定の合格にしない。pp2048 後の 128 token についても、温度 0 の target-only と DFly の全列を比較する。

## Q2　CPU 対照と 3.5 bpw への移行条件

ユーザー指定により、初期目安 0.1 と、同一入力・同一配置の CPU 積和順序対照の変動を分けて記録する。固定配置 7332 / 6830 / 6830 expert、入力 ID 120000・802・15063・2576 では、**EP=3 の最終 logits の相対 L2 は 0.133337、CPU reorder 対照は 0.139013、温度 0 の通常生成と DFly の greedy 列は 24 token 完全一致**。次 token は CPU と EP=3 ともに 12286、KL（CPU から EP=3）は 0.05572。対照は同じ packed tensor を用い、CPU 上だけで mul24 内積・iHC・正規化・router の FP32 積和順序を変えたもの（`build-hy4/recalc-rdna.py`、`shelf-cpu-stable-rdna`）。変更した順序を再実行するため check_chain.py に --rounding-control を用意した。相対 L2 が 0.1 を超えても、CPU 対照変動以下なら棄却理由としない。

`ep3-19qprof9` の固定配置・chunk 1024 は pp2048 **674.80 tok/s**、decode **50.82 tok/s**（初回を除く 3 回の中央値）。pp2048 に続く 128 token も通常生成と DFly がすべて一致し、正の温度の同一 seed 結果も再現した。候補整理の後、採用版では 256 token の照合を実施する。DFly2 の量子化、Engram FP8 化、2Mi 実予約はいずれも完了しており、長い列の照合と候補整理が済んでから 3.5 bpw へ進む。

## P5　候補採否と切り替えコードの撤去

固定配置、同一バイナリ、同一入力・chunk 1024・seed による比較。時刻・logits・実際の全生成列は `ep3-19qprof9/ablation/` に残した。速度は初回を除く 3 回の中央値。下表の列一致は 4 token 入力から生成した 24 token と全有効版の比較を表す。

| 無効にした候補 | CPU logits 相対 L2 | 次 ID | 全有効版と列一致 | pp2048 tok/s | decode tok/s |
|---|---:|---:|---|---:|---:|
| 全候補有効（基準） | 0.13334 | 12286 | 一致 | 674.80 | 50.82 |
| WMMA | 0.14100 | 12286 | 不一致 | 47.73 | 17.76 |
| DOT4 | 0.13334 | 12286 | 一致 | 674.45 | 28.61 |
| ABSORB | 0.15460 | 12286 | 不一致 | 677.77 | 28.33 |
| GROUP_ROWS | 0.13334 | 12286 | 一致 | 343.83 | 41.65 |

4 候補とも採用した。R9700 の射影・連続範囲の prefill は WMMA、短い行・疎な選択は吸収形 MLA。routed expert は少数行で dot4、8 行以上では expert ごとに最大 16 行をまとめて R9700 は WMMA、V620 は LDS に展開したタイルを dot4 で共有する。候補フラグを 4 個とも外し、比較専用だった scalar 射影・attention・吸収形 kernel と試験用スイッチを削除した。legacy は共通 arena と QSA を既存呼び出しから検査する用途があるため維持する。採用した配置表は `hy4-flash-pilot/shelf-placement/owners-4bpw.json` に保存。候補撤去後のビルド・256 token 列・速度の再確認が次の受入である。

## 毎時レビュー第 5 回への対応

採用版を新規ディレクトリでビルドし、CPU テストを再確認した。256 token の検査は対象・DFly 間の照合に加え、P5 の比較に使った全候補有効バイナリでも同じ入力・配置・seed で実行し、採用版の全列と比較する。旧バイナリの SHA-256 は P5 の設定記録と照合してから実行する。rounding_control.py、配置表、研究中の一時スクリプトの継続用途はこの移行判定後に記録する。

## G3　4 bpw の最終受入と 3.5 bpw への移行

新規ビルドと CTest が通過。採用版実行形式の SHA-256 は a7365f79d3684159121c4cef1d162c2b7757ed6874c9657bdfe9113c357763db。pp2048 は **676.62 tok/s**、127 token の decode は **50.65 tok/s**（初回を除く 3 回の中央値）。2Mi 予約、CPU / EP=3 top-1 12286、相対 L2 0.133337、対照 0.139013、KL 0.05572 を確認。採用版の通常生成、採用版の DFly、切替撤去前の全候補有効版（SHA-256 783062b2355dc5548f03802a486eed999adab3da448ec534aa7f09c5ac217580）の 3 列は **256/256 一致**。正の温度でも同一 seed の結果を再現した。実測記録は `hy4-flash-pilot/shelf-qualification/ep3-9ao7j7r7/{results,cleanup-comparison}.json`。

第 5 回レビューの衛生確認。rounding_control.py は採用版の CPU 変動の再現と 3.5 bpw の同条件検査に使用するため維持する。公式の --rounding-control で一時実装と同じ logits を再現した。配置表も通常実行と比較の入力として SHA-256 付きで `hy4-flash-pilot/shelf-placement/` に保持する。役割を終えた未追跡の recalc-rdna.py などの一時実装は除去し、測定ログは残した。

## M2　3.5 bpw 変換の開始

4 bpw の G3 完了後に V620 2 台を排他し、BF16 原本と同じ 128×1024 校正の hidden / router 選択から Hessian を再計算している。層ごとに利用頻度の低い expert を 3 bit にし、合計 10,931 本を選んだ。残り 10,061 本と本体の通常線形は 4 bit、共有 head は 6 bit、DFly は G2 の 4 bit を共用する。符号のみの目標平均は 3.499996 bpw。最終出力先は `hy4-flash-pilot/shelf-mixed/Hy4-flash-exl3-3.5bpw-mul24`。shard 単位で署名・SHA-256 を照合して再開でき、元の 4 bpw checkpoint を変更しない。

## M3　3/4 bit 混在の読込と実層

各 expert の 3 射影の幅を実 trellis から検証し、3/4 bit 混在では実バイト数で再配置する。R9700 WMMA と V620 dot4 は行または expert グループのビット幅に合わせて復号する。既存の 4 bit 専用経路は定数幅のままビルドする。3.5 bpw 版は元の校正署名で DFly と照合し、FP8 Engram の符号・スケールを checkpoint 自体に格納する。CPU loader と照合も混在幅に対応した。

変換済みの層 1 は 3 bit expert 267 本。校正先頭 16 token の元の 256 選択には 3 bit が 71 件含まれた。R9700 で同じ mixed 層を実行し、CPU 基準への相対 L2 は 16 行一括 0.004823、逐次 1 行 0.005646。expert 4 の実 3 bit gate の grouped 6 行は相対 L2 0.0000283。両 ISA をビルドし、既存 4 bit + DFly の inspect と CPU CTest も通過した。全校正で 3 bit 化する expert の選択は 19,977,959 / 85,983,232 件（23.23%）。原本の routing から独立に再計算した選択リストの SHA-256 も変換計画と一致した。

3.5 bpw の shard を検証しながら同じ 4 token の CPU 連続照合を追跡中。組立後は `build-hy4/mixed-accept.log` の手順で、同じ配置・pp2048・256 token の greedy 照合、CPU reorder 対照、EP=3 の実予約・速度を記録する。

## M4　混在幅の内積と再構成

同一 launch に実 3 bit の 8 行と 4 bit の 16 行を並べ、R9700 で grouped WMMA と LDS/dot4 を両方照合した。どちらも CPU 相対 L2 は 0.00000670。層 1 expert 4 の元 BF16 に対する再構成相対 L2 は 4 bit → 3 bit で gate 0.09190 → 0.18110、up 0.09198 → 0.18120、down 0.07586 → 0.14987。低ビット化する重みの誤差であり、生成品質とは区別する。再現用の 2 台 lock の起動例を計画書に追記した。

## M5　混在版の受入準備と変換済み shard の追跡

04:12 JST 時点で 197 / 656 shard、3,279 / 10,931 expert を保存し、12 個の MoE 層ですべての shard がそろった。CPU は層 11 まで連続照合し、同層の BF16 校正出力に対する相対 L2 は 0.077759。正規の build-hy4 でも両 ISA のビルドと CPU CTest が通った。4 bpw の 1,161 shard の header を独立に集計し、decoder 符号平均 4.0、補助 tensor 込み 4.020704 bpw、未量子化線形も含めて 4.022769 bpw を再現した。3.5 bpw の完成後にも同じ集計を行う。

受入では実際の生成列長が 256、速度測定の decode 区間が 127 token であることを検査する。混在幅対応後の実行形式でも、4 bpw で全候補有効版の 256 token を再現するかを照合する。fork の既定構成の起動例は README、4 bpw の基準値は [性能記録](../../perf/current.md#hy4-flash-専用-fork-の-ep3) に反映した。

## 毎時レビュー第 6 回と変換の中間点

追加の指摘はなかった。04:42 JST に 328 / 656 shard、5,476 / 10,931 expert に到達した。層 20 までの全 shard がそろい、同じ 4 token の CPU 連続照合も層 20 まで完了した。BF16 校正出力に対する同層の相対 L2 は 0.033874。変換後は同じ配置表と測定条件で EP=3 を測る。混在幅対応後のバイナリでの 4 bpw・256 token 比較も待機している。

## 毎時レビュー第 7 回と層 30 の比較

追加の指摘はなかった。05:19 JST の 8,257 / 10,931 expert は marker 495 / 656 個に対応する。レビュー記載の 328 個は 04:42 JST 時点の値。05:24 JST には 516 個、8,600 expert が保存された。同一の入力 ID・配置で保存した CPU の中間出力を比較した。

| 層 | 4 bpw の BF16 比相対 L2 | 3.5 bpw の BF16 比相対 L2 |
|---:|---:|---:|
| 10 | 0.079850 | 0.079457 |
| 20 | 0.033091 | 0.033874 |
| 30 | 0.132645 | 0.145328 |

比較対象は同じ校正列の先頭 4 token。全層通過後の logits、DFly の生成列、速度、予約量は組立完了後に測定する。
