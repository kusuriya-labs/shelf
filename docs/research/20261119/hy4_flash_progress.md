# Hy4-flash の進捗

origin/main の 14a23730 から専用 worktree を作成した。原本の config と量子化側の組立仕様を確認した。本体と head の変換完了待ち。組立は Engram を BF16 のまま保持し、DFly を除外するため、どちらも別途処理する。GPU を使わない作業から着手する。

## E1　Engram の保存形式

変換器を追加した。原実装の BF16 スケール・E4M3FN 丸め順序に従い、1 行 96 byte の符号と 2 byte のスケールを safetensors に格納する。7 行のテストでゼロ行と最終端数チャンクを含む復元値が一致した。原本の config 上の層番号は 1 始まりで、tensor 名は 1、11、21、31。原本からの 4 表の変換を CPU で開始した。出力先は hy4-flash-pilot/engram-e4m3fn。本体の progress は確認時点で層 26 まで。量子化スクリプト上の最終 report の位置は rocm-exl3-rdna2/models/Hy4-flash-exl3-4bpw-mul24/conversion_report.json。
