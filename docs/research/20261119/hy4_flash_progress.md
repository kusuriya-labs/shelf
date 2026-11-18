# Hy4-flash の進捗

origin/main の 14a23730 から専用 worktree を作成した。原本の config と量子化側の組立仕様を確認した。本体と head の変換完了待ち。組立は Engram を BF16 のまま保持し、DFly を除外するため、どちらも別途処理する。GPU を使わない作業から着手する。

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
