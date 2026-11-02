# EAGLE-4 Phase A Gate 記録

- ID: EAGLE4-PHASE-A-GATE
- 実施日: 2026-10-22（再測定）
- 判定: PASS（vanilla 比 1.79×、parity 完全一致）。DFlash2 比較は本環境で実施不能。


### Phase A Gate の判定（2026-10-22 再測定）

Phase A Gate は **PASS** である。測定構成は `--tp 2`（FP8 target は単 GPU では
draft 自前 lm_head と同居できないため）、比較対象は vanilla である
（DFlash2 比較は本環境で実施不能。下記参照）。

**target の復旧と受理率0の原因**（2 件）:

- 旧判定の直接原因は本 branch の `output_gate_type` 変更にあった。attention 出力ゲートは
  常に sigmoid であり、`output_gate_type` は GDN の z ゲート活性を決める
  （`docs/developer/qwen35.md` §1.1-1.2 に正誤した）。target 出力は main の build と
  トークン完全一致する。rebased build で `test_gemm_fp8_optimized` も通る。
- candidate head の再評価 head が不一致だった。`--eagle4-lm-head target`（既定）では
  rough-score の factor（draft の head の SVD）と再評価に使う head（target の head）が
  別物になる。公式実装は draft checkpoint 自前の `lm_head` を使う
  （`eagle_harness` の `LlamaForCausalLMEagle4.set_embed_and_head` は target から
  embed のみ共有し head は draft 保持）。`--eagle4-lm-head draft` で
  `EAGLE4_MEAN_ACCEPTED=0.000` が 3.400 になった。

**測定構成**（`docs/perf/methodology.md` 準拠）:

- `shelf-compute --model-dir models/Qwen3.8-27B-FP8
  --eagle4-model-dir models/Qwen3.8-27B-EAGLE4 --eagle4-drafts 7
  --eagle4-lm-head draft --device 1 --tp 2 --prefix-cache 0 --max-seq-len 256`、greedy。
- GPU1/2（隔離。GPU4 の他 workload と非共有）、warmup 2 run 後に baseline/candidate を
  交互に 5 run、p50 中央値。run 間 spread は vanilla 1.8% / EAGLE-4 0.2%（閾値 30% 以内）。
- workload は 6-token prompt から 128 token 生成（decode 128 token の完全一致確認）。

**正しさ（PASS）**: 同一 prompt・同一 `--max-new-tokens` で vanilla と EAGLE-4 の
`GENERATED_IDS` が逐トークン一致した。

| prompt | 生成 token | drafts | 一致 |
| --- | --- | --- | --- |
| chat template（59 token prompt） | 20 | 1 / 3 / 7 | ○ |
| 6-token raw completion | 128 | 7（12 run） | ○ |
| 1000-token 長文脈 | 64 | 7 / 1 | ○（部分 reject 含む。mean accept 2.94 / 0.80） |

**性能（PASS、5-run p50）**:

| arm | decode tok/s p50 | run 間 spread |
| --- | --- | --- |
| vanilla | 32.15 | 1.8% |
| EAGLE-4（drafts 7） | 57.55 | 0.2% |

speedup = **1.79×**。1 round は約 51.3 ms（draft GPU 11.3 ms、verify 37.5 ms、
GDN snapshot/restore は無視できるほど小さい）。受理数は prompt 依存で、
raw 文 1.953/7（emitted 2.953/round・43 round）、chat 3.400/7、1000-token 文脈 2.938/7。

**DFlash2 比較は実施不能**: DFlash2 draft は 3.58 GiB で FP8 target（27.45 GiB）と
単 GPU に同居せず memory plan が失敗し、`--tp 2` の dflash2 配線も未実装である。
計画の「DFlash2 未満なら置換せず停止」は、ユーザー判断で「vanilla 未満なら停止」に
読み替えて判定した。EAGLE-4 は vanilla を 1.79× 上回るため **Phase B へ進む**。
