# EAGLE-4 Phase D Gate（batch 化の採否）

- ID: EAGLE4-PHASE-D-GATE
- 実施日: 2026-10-23
- 環境: TP2（device 2/3、`SHELF_MODEL_HOST_SOCKET` の2台 host、
  Qwen3.8-27B-FP8 + Qwen3.8-27B-EAGLE4 / Qwen3.8-27B-DFlash2）
- 方法: `tools/dgate.py`（single-shot の参照取得 + `--serve-stdio` の
  N 並列 generate、5-run 交互 p50。測定原則は docs/perf/methodology.md に従う）

## 判定

| 項目 | 結果 | 根拠 |
| --- | --- | --- |
| 正しさ（concurrent=4 == single） | **ACCEPT** | eagle4 chain / eagle4 tree(W8/D4) / dflash2 の3モード × 4プロンプト（chat / chat先頭30 / 既定 / chatシフト）= 12/12 完全一致、error なし |
| 効率（DFlash2 batch に負けない） | **REJECT** | 下表 |
| **総合** | **REJECT → EAGLE-4 の batch 経路は不採用。EAGLE-4 は single-request のまま** | D4（DFlash2 の TP2）は比較基盤として独立に採用 |

## 性能（5-run p50、64 token/req、TP2、model-host）

| arm | N=2 tok/s | spread | N=4 tok/s | spread | vs single×N |
| --- | --- | --- | --- | --- | --- |
| single（eagle4 chain、基準） | 49.04 | — | — | — | 100% |
| eagle4 chain batch | 67.13 | 10.7% | 74.91 | 13.5% | 68.4% / 38.2% |
| eagle4 tree(W8/D4) batch | 73.99 | 0.2% | 94.98 | 10.1% | 75.4% / 48.4% |
| dflash2 batch | 93.26 | 5.2% | 126.29 | 5.1% | 95.1% / 64.4% |

## 観測された知見

- **tree は batch で chain を上回る**（N=2: 74.0 vs 67.1、N=4: 95.0 vs 74.9）。
  single-request で tree が chain に負けた Phase B から符号が反転しており、
  「continuous batching で tree が効く」という仮説は測定で支持された。
- eagle4 chain の N=4 効率（38%）が dflash2（64%）より著しく低い。
  仮説（未検証）: dflash2 は `dflash2_propose_batch` で draft を1バッチに束ねるが、
  eagle4 の batch は slot ごとに draft を直列実行しているため。
- `test_dflash2_tp_e2e`（optional）の TP1 段は、FP8 target27.45 + dflash3.58 の
  resident が単カード 31.86GiB に同居せず OOM する（docs 記載の同居不可と同じ壁）。

## 既知の測定上の注意

- single の49.04 tok/s は「既定プロンプト（248041,77091）・64 token・
  --eagle4-lm-head draft・model-host」の値であり、docs/perf/current.md の
  他プロンプト条件の数値とは比較しない。
