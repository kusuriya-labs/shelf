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
