# Hy4-Flash 疎層バッチ化・SWA `unpack_quant` 融合・`c=32` 拡張・3-bit/混在高速パス（2026-11-24）

ベースコミット: `4651499d`（作業ブランチ `perf/hy4-c32-sparse-mixed`、単一 Radeon AI PRO R9700 `gfx1201`・ROCm 10.2 上で実測）。
測定原則は [docs/perf/methodology.md](../../perf/methodology.md) に従う。

---

## 実験一覧

| ID | 対象 | 変更概要 | 判定 |
|---|---|---|---|
| EXP-01 | `attention_batch` 疎層 (`pos > 2048`) | スロットごとの直列 `attention_inner` + `hipMemcpyAsync` ループを廃止し、複数疎スロットをまとめてバッチ実行（`index_scores_sparse_batch` / `index_topk_sparse_batch` / `build_sparse_indices_batch` / `unpack_quant_sparse_batch` / `unpack_rot_sparse_batch` / `shelf_hy4_mla_absorbed_sparse_batch`）へ統合 | ACCEPT |
| EXP-02 | SWA 30 層・連続経路 | `attention_inner` の `unpack_quant` を全経路（`chunk_contiguous=true` / `draft` / `sparse`）へ拡張し、`attention_batch` 連続経路へ `unpack_batch_quant` + `prequant` を導入して中間 BF16 往復を排除（BF16 RNE 丸め一致維持） | ACCEPT |
| EXP-03 | `c=32` / `max_rows=192` 拡張 | `Residency` / `Hy4GroupedPlan` / `BatchGraphParams` / `BatchGraphInput` / `GraphKey` / `shelf_hy4_head*` / `wmma.hip` / `moe.hip` / `hy4_compute` / `shelf_server.py` / `placement.py` の上限を `16 -> 32` スロット・`96 -> 192` 行へ拡張 | ACCEPT |
| EXP-04 | 3〜4 bpw 混在 / `bits == 3` | `moe.hip` の `!op.gpu.model.mixed` ガードを撤去し実ビット幅で判定。`wmma.hip` / `linear.hip` / `prefill.hip` で `bits == 3`（融合 GEMM・WMMA・shared pair・recon）を開放し、`placement.py` を実 bpw 別の正確なバイト数計算に対応 | ACCEPT |

---

## 前後比較ベンチマーク（`4 bpw`, 単一 R9700, EP=1）

### 1. 長文コンテキスト別 `ctxbench`（`B=1`: `ctx=128/2048` 12 ステップ中央値・3 試行、負数は Graph オン／オフ時の差；`B=16/32`: `ctx=2100`, 32 トークン／スロット）

| 条件 | 指標 | Before (`4651499d`) | After | 変化 |
|---|---|---|---|---|
| `CTX=128 B=1 draft=0 Graph OFF` | trial 1 / trial 2 `step_ms` | 27.79244 / 27.71495 | 27.65115 / 27.60891 | -0.14ms (-0.5%) |
| `CTX=128 B=1 draft=0 Graph ON` | 中央値 `step_ms` | 26.31301 | 25.85837 | -0.45ms (-1.7%) |
| `CTX=2048 B=1 draft=0 Graph OFF` | trial 1 / trial 2 `step_ms` | 34.09211 / 34.04657 | 32.67068 / 32.74611 | -1.42ms (-4.2%) |
| `CTX=2048 B=1 draft=0 Graph ON` | 中央値 `step_ms` | 31.44717 | 30.90745 | -0.54ms (-1.7%) |
| `CTX=128 B=1 draft=1 Graph OFF` | trial 1 / trial 2 `step_ms` | 55.53288 / 55.49611 | 55.58507 / 55.58309 | +0.05ms (同等) |
| `CTX=128 B=1 draft=1 Graph ON` | 中央値 `step_ms` | 57.45617 | 57.32046 | -0.14ms (-0.2%) |
| `CTX=2048 B=1 draft=1 Graph OFF` | trial 1 / trial 2 `step_ms` | 63.73898 / 63.58072 | 62.88006 / 62.77214 | -0.86ms (-1.3%) |
| `CTX=2048 B=1 draft=1 Graph ON` | 中央値 `step_ms` | 64.46812 | 64.00294 | -0.47ms (-0.7%) |
| `CTX=2100 B=16 draft=0 Graph OFF` | `batch_decode_s` / `agg_tok_s` | 3.800467s / 130.51 tok/s | 2.057396s / 241.08 tok/s | **+84.7%** |
| `CTX=2100 B=16 draft=0 Graph ON` | `batch_decode_s` / `agg_tok_s` | 3.897796s / 127.25 tok/s | 2.112250s / 234.82 tok/s | **+84.5%** |
| `CTX=2100 B=16 draft=1 Graph OFF` | `batch_decode_s` / `agg_tok_s` | 10.696064s / 46.37 tok/s | 7.016853s / 70.69 tok/s | **+52.4%** |
| `CTX=2100 B=16 draft=1 Graph ON` | `batch_decode_s` / `agg_tok_s` | 6.566557s / 75.53 tok/s | 4.691258s / 105.73 tok/s | **+40.0%** |
| `CTX=2100 B=32 draft=0 Graph OFF` | `batch_decode_s` / `agg_tok_s` | （未対応） | 2.738916s / 362.19 tok/s | 新規 (`c=32`) |
| `CTX=2100 B=32 draft=0 Graph ON` | `batch_decode_s` / `agg_tok_s` | （未対応） | 2.731732s / **363.14 tok/s** | 新規 (`c=32`) |
| `CTX=2100 B=32 draft=1 Graph OFF` | `batch_decode_s` / `agg_tok_s` | （未対応） | 8.652452s / 114.65 tok/s | 新規 (`c=32`) |
| `CTX=2100 B=32 draft=1 Graph ON` | `batch_decode_s` / `agg_tok_s` | （未対応） | 8.698590s / 114.04 tok/s | 新規 (`c=32`) |

### 2. 短文 `stepbench`（`ctx=16`, 5 試行中央値 ms）

| 条件 | Graph OFF Before -> After | Graph ON Before -> After |
|---|---|---|
| `B=1 draft=1` | 55.59720 -> 55.23948 ms | 56.87066 -> 56.40490 ms |
| `B=1 draft=0` | 26.68261 -> 26.44865 ms | 25.49150 -> 25.03204 ms |
| `B=16 draft=1` | 96.19575 -> 95.51507 ms | 95.90117 -> 94.58164 ms |
| `B=16 draft=0` | 40.74341 -> 40.41734 ms | 39.91369 -> 38.86307 ms |
| `B=32 draft=1` | （未対応） -> 165.62282 ms | （未対応） -> 167.16721 ms |
| `B=32 draft=0` | （未対応） -> 45.99123 ms | （未対応） -> 45.73950 ms |

### 3. `microbench`（単一演算カーネル、`output_hash` 完全一致確認済み）

`rows=1, 6, 16, 96` の全カーネル（`linear`, `fixed_ffn`, `mla_qb`, `mla_o`, `lm_head`, `lm_head_argmax`, `experts`）で出力ハッシュ一致・性能劣化なし。
`c=32` 向けに追加した行数の結果:
- `rows=32`: `linear=23.193us`, `fixed_ffn=70.743us`, `mla_qb=63.375us`, `mla_o=142.304us`, `lm_head=574.464us`, `lm_head_argmax=555.968us` (`32/32`), `experts=3029.435us` (`601.227 GB/s`)
- `rows=192`: `linear=67.452us`, `fixed_ffn=197.123us`, `mla_qb=161.239us`, `mla_o=445.683us`, `lm_head=2751.128us`, `lm_head_argmax=2659.901us` (`192/192`), `experts=3330.284us` (`546.913 GB/s`)

---

## 3.5 bpw 混在 (`body35`) 実機検証結果

1. **`bits == 3` 線形カーネル精度・速度（`bits3bench`・CPU `mul24` 参照比）**:
   - `rows=1`: `us=15.651` (`56.923 GB/s`), `rel_l2=0.001726`
   - `rows=6`: `us=24.592` (`36.226 GB/s`), `rel_l2=0.001674`
   - `rows=16`: `us=43.884` (`20.301 GB/s`), `rel_l2=0.001672`
   - `rows=32`: `us=46.351` (`19.220 GB/s`), `rel_l2=0.001668`
   - `rows=96`: `us=47.763` (`18.652 GB/s`), `rel_l2=0.001654`
   - `rows=192`: `us=50.437` (`17.663 GB/s`), `rel_l2=0.001654`
   - `rows=256`（`recon` + WMMA prefill 経路）: `rel_l2=0.001654`
2. **混在モデルでの `shared_gate_up` 解放**:
   - `rows=1, 6, 16` で非融合 `op.linear` 経路と `rel_l2=0.000000` で一致。
3. **混在 3/4-bit MoE (`moebench`) と E2E `stepbench` (`body35`, Layer-1 expert 共有, EP=1)**:
   - `moebench`: `rows=1: 244.756us`, `rows=6: 814.829us`, `rows=16: 1121.890us`, `rows=32: 1398.843us`, `rows=96: 1985.642us`, `rows=192: 2711.618us`
   - `stepbench draft=1`: `B=1: 57.58ms (OFF) / 58.78ms (ON)`, `B=16: 98.68ms (OFF) / 97.75ms (ON)`, `B=32: 172.49ms (OFF) / 173.79ms (ON)`（Graph・Eager 間トークン完全一致）
4. **容量見積り（`placement.py`）**:
   - `body35` + `draft`, `--tokens 8192 --max-concurrent-requests 32 --capacity 32 32 32` にて 3 GPU 配置（`[6515, 7238, 7239]`, `free_after=[4.14 GB, 10.60 GB, 10.60 GB]`）を算出可能。単体テスト `tests/unit/test_hy4_flash.py` 10/10 通過。

---

## 検証結果サマリー

- CPU CTest (`-L "cpu" -LE "optional"`): **133/133 Passed**
- Python 単体テスト (`tests/unit/test_hy4_flash.py`): **10/10 Passed**
- GPU 単体テスト (`test_hy4_reduce`, `test_hy4_peer_transfer`): **PASS**
- `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph $BODY $DRAFT`: **EP=1, 3, 4 PASS**
- `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph $BODY $DRAFT batch`: **EP=1..4 (`B=2..32`, `DFly2 ON/OFF`, `Graph ON/OFF`), `session=pass`, `mixed_sparse=pass`, `c32_sparse=pass` PASS**
- `SHELF_HY4_CHECK_EVENT_REUSE=1 test_hy4_graph $BODY $DRAFT transport`: **EP=3, 4 (`push/peer/host`, `draft=0/1`, `B=1..16`) PASS**
- `test_hy4_graph $BODY35 bits3bench` / `$BODY35 $DRAFT stepbench {1,16,32} 1`: **PASS**
