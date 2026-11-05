# MoE tensor parallel は expert 軸で分割し、routing は model 全体で行う

実験 ID `MOE-TP`、結論 **ACCEPT（expert 軸分割 + 全局 routing + combine で collect）**。

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8・
moe_intermediate 512・shared_expert_intermediate 512）を TP=2 で動かすための分割方針。
現状の実測は合成 fixture（experts 4・layers 2）までで、実 model の E2E と性能は未測定。

確定した contract は [../../developer/tensor_parallel_execution.md](../../developer/tensor_parallel_execution.md) §11-1 と
[../../developer/qwen35moe.md](../../developer/qwen35moe.md) §3 / §6 に置く。ここでは採否の理由だけを残す。

## 1. 分割軸の候補と採否

dense FFN は column / row parallel（gate / up を intermediate 軸、down を K 軸）で分割できる。
routed expert 平面 `[experts][rows][cols]` に対しては三つの軸が考えられた。

| 案 | 判断 | 理由 |
| --- | --- | --- |
| expert 軸（rank が expert の contiguous 範囲を持つ） | **採用** | 1 expert の layout（rows / cols / k_padded / native batch stride / preshuffle）がまったく変わらない。rank 平面を global 順に並べた byte 列は単一 GPU の平面と一致し、grouped GEMM も activation 平面も per-expert の式をそのまま使う。checkpoint の expert tensor は expert-major で連続なので、rank が読む範囲も連続（I/O も半分） |
| expert 平面の N 軸（rows）を matrix partition で割る | 不採用 | MXFP4 の rank-3 canonical partition は `materialize_rank_local_canonical` が `unsupported` で、preshuffle 済みの block layout を軸で割ると grouped GEMM の fragment order 前提が壊れる。grouped kernel は expert 単位で batch stride を読むので、rows を割っても expert 数は減らず dispatch 数も減らない |
| top_k を割る（1 token の選択を rank に割り振る） | 不採用 | 選択は global に決める必要がある（§2）。expert 単位ではなく token 単位で割ると、1 expert に対する行が rank をまたぎ、grouped GEMM の group が expert をまたがない契約と衝突する |

## 2. router を分割しない理由（ここが設計の核心）

rank が自分の expert 範囲だけの logits で top-k を選ぶと、各 rank の選択と再正規化が
「その rank が見えた expert 内」の条件付きになる。例えば experts 0..3 / top_k 2 で
logits が `(3, 2, 1, 0)` のとき、全体 top-k は `{0,1}` で weight は `(0.6, 0.4)`。
範囲を分割して各 rank が自範囲で top-k を取ると、rank0 は `{0,1}`、rank1 は `{2,3}` を選び、
combine の合計は元の結果にならない（ expert 2, 3 が重み付きで混ざってしまう）。

そのため **softmax・top-k・再正規化は logits 幅（model 全体の expert 数）で行い、plan だけを
局所化する**。この形を contract として固定した実測が `test_moe_router` の split case である。
unsplit と 2 rank の `sel_weight` が全 slot で bit 一致し、各 slot はちょうど一方の rank でのみ
group row を持つ。所有外の slot は `sel_expert` / `slot_row_of` とも -1 で、combine はこれを
加算しないので、自然に部分和ができる。

router weight（`mlp.gate.weight`、256×2048 BF16 で 1 MB/層）は replicated のまま置いた。
40 層で 40 MB / rank で、分割して得られる memory より、barrier が 1 回増える費用のほうが大きい。

## 3. shared expert は intermediate 軸で分割

shared expert（intermediate 512）は dense FFN と同じ column / row parallel で割った。
候補との比較:

| 案 | 判断 | 理由 |
| --- | --- | --- |
| intermediate 軸分割（gate / up axis 0、down axis 1） | **採用** | combine の出力が 部分和 になるという §2 の形にそのまま収まる。barrier は FFN あたり 1 回のまま増えない。weight も compute も半分。partition 済み shared expert の invariant は weight load 側に実装済み |
| 全 rank で複製 | 不採用 | weight 複製（1 層 3 MB × 40）と余剰 compute が残る。memory も減らない |
| rank0 のみが計算 | 不採用 | compute は 1 回で済むが、rank 間で graph が非対称になり、barrier の arrival 条件と footprint・plan scratch の rank 間比較が複雑化する。contract として説明しにくい |

shared expert gate（`mlp.shared_expert_gate.weight`、1×2048）は replicated。これは token 毎の
スカラーで、rank 間の部分和に依存しないため、どの rank でも同じ値になる。

## 4. 実測（合成 fixture・2 GPU）

`test_qwen35moe_tp_execution`（gpu2;optional）。fixture は experts 4・layers 2（GDN + full
attention）・top_k 2・hidden 256・inter 128・kv heads 2・vocab 1024。prompt 16 token +
decode 2 step、`max_batch_rows 16`。transport は `hip-p2p`（peer access 有効）。

| 比較 | 値 |
| --- | --- |
| TP=1 greedy token | `1 5 6` |
| TP=2 rank0 / rank1 greedy token | `1 5 6` / `1 5 6`（一致） |
| hidden 行（row0）の TP=1 に対する相対差 | 0.00033 |
| logits 行（row0）の TP=1 に対する相対差 | 0.0135 |
| barrier 後の logits の rank 間差 | bit 一致（0） |
| rank の expert 平面 depth / shared 幅 / KV heads / router logits | 2 / 64 / 1 / 4（experts 4・shared 128・kv heads 2） |

logits の 1.35 % は dense TP と同じ帯（dense は層別 0.6〜2.7 % を
[../../developer/tensor_parallel_execution.md](../../developer/tensor_parallel_execution.md) §10 に記録済み）。
barrier が部分和を f32 で累積し bf16 へ 1 度だけ丸める契約のため、rank 間では bit 一致する。

weight 側は `test_qwen35moe_model_load` が固定する: rank 平面の byte 列は単一 GPU load の
対応する expert と一致し、rank の weights は whole-model invariant を通らない。

## 5. 未測定・未配線

- 実 model（Qwen3.6-35B-A3B）での TP=2 の E2E 数値と性能は未測定。`TpRankRuntime` と
  `TpCoordinator` は dense の rank load しか行わないので、`shelf-compute --tensor-parallel` と
  `shelf-bench pp / quality --tensor-parallel` は grouped expert checkpoint を起動時に拒否する。
  multi-rank runtime の MoE 配線（arena 予算、KV / GDN pool の local geometry、
  `moe_executor_grouped_bytes` を使った reserve）が残作業
- MoE MTP block は分割しない（dense と同じ。`--mtp` と `--tensor-parallel` の併用は compute CLI が拒否）
- expert 配分は均等な contiguous 範囲のみ。負荷に応じた expert 配置（真の EP scheduler）は非目標
- 性能主張は実 model の TP=1 vs TP=2 実測まで行わない（`docs/perf/current.md` には書かない）
