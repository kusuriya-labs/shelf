# EXL3-INT8 mul24 prefill の品質

Qwen3.8-27B、cases-v1 の 8 case・2,632 位置、R9700、ROCm 10.2、BF16 KV。基準は llama.cpp BF16・FP16 KV の logits。温度 1、KL(BF16 ‖ 候補)、有効語彙 248,077 ID。A は gfx1201 の既定 prefill、B は `SHELF_EXL3_MUL24_PREFILL_AB=B`。先頭最大 256 token を prefill し、残りを逐次 decode した。

| 量子化 | 経路 | 平均 KLD | top-1 一致 |
| --- | --- | ---: | ---: |
| 4bpw | B | 0.020745 | 94.8708% |
| 4bpw | A | 0.021176 | 94.7568% |
| 3.05bpw | B | 0.059129 | 91.3374% |
| 3.05bpw | A | 0.059471 | 91.1474% |

4bpw structured case の A/B KLD 比は 1.246（0.016366 / 0.013136）。case 別の差は一様ではない。A と B の経路差は chunk・入力に依存する。

4bpw 長文の BF16 基準比較では、4,097 token の 2 case・末尾 514 位置の平均 KLD / top-1 は B 0.004584 / 98.4436%、A 0.004577 / 99.0272%。反復 8k 入力の後続 64 位置は B 0.743497 / 87.50%、A 0.491307 / 89.0625%。8k は単一の反復入力であり一般化できない。BF16 基準の KV は FP16、候補は BF16 で、engine・chunk も異なるため、差は量子化誤差だけを示さない。

A の algorithm cache は要求 workspace サイズも保存するが、**非ゼロ workspace を要する algorithm の cache 再利用は未検証**。B と A のいずれも prefill の分割により出力が bit 単位で変化し得る。通常生成と DFlash2 の不一致は B にも既存であり、ユーザーの採否方針では BF16 比の KLD / top-1 を判断基準とする。詳細な評価と一覧は [A 経路の評価](quality-bf16/a-path-assessment.md)、[BF16 基準の測定結果](quality-bf16/results.md)。

## 既定 A と B の動作確認

2026-11-20、ROCm 10.2、R9700、Qwen3.8-27B EXL3-INT8 4bpw、BF16 KV。各条件 1 測定、pp と tg は process 内 warmup 1 回。c=16 は 16 要求・各 2048 token を同時投入し、prefill は最後の最初の token までの壁時計、decode は全要求が活動中の合計速度。DFlash2 は shell 40 token、drafts 7、tree budget auto。c=16 と DFlash2 の各経路は別 process、交互実行ではなく、次の値は参考値。

| 指標 | A 既定 | B | 単位 |
| --- | ---: | ---: | --- |
| pp2048 | 2352.33 | 1134.98 | tok/s |
| c=1 decode | 32.47 | 32.24 | tok/s |
| c=16 prefill | 1822.88 | 927.46 | tok/s |
| c=16 decode | 273.91 | 271.72 | tok/s |
| DFlash2 decode | 78.73 | 78.98 | tok/s |

`test_exl3_mul24_linear`、`test_gemm_lt`、`test_hipblaslt_abi`、GDN・dispatch を含む required CTest 17 件は A/B とも全件通過した。LFS の 2 fixture が pointer の環境では GDN reference が失敗するため、実体を取得して SHA-256 を照合した状態で全 17 件を再実行し、終了後に pointer を復元した。

INT8 heuristic のみを `HIPBLAS_STATUS_NOT_SUPPORTED` とする障害注入では、A は B に戻り、code 入力 2048 token の後続 64 行の dump が B と bit 一致した。shelf の fallback 通知は 1 行。rocRoller 指定時も同条件の B と一致し、通知は 1 行だった。共有 `libhipblaslt.so.1` 自体は実行ファイルの直接依存であり、ライブラリが欠落すると起動前に失敗する。
