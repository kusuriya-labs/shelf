# Multi-GPU Execution

PP / EP / TP の current contract である。weight storage 側の partition
（`TensorPartitionDesc` / manifest / materialization）は
[tensor_partition.md](tensor_partition.md) が正本で、本ドキュメントは runtime の
座標・所有構造・collective・比較条件を扱う。

## 1. Topology

3 軸は `PP > EP > TP` の順に大きい。stage 内の device は次の rank 順で並べる。

```text
rank = (pp * ep + e) * tp + t
```

`--device` は `pipeline_parallel * expert_parallel * tensor_parallel` 個を
PP → EP → TP の順で指定する。8 GPU で PP=2 / EP=2 / TP=2 なら次のようになる。

```text
GPU   0 1 2 3 4 5 6 7
PP    0 0 0 0 1 1 1 1
EP    0 0 1 1 0 0 1 1
TP    0 1 0 1 0 1 0 1
```

`ParallelTopology`（`include/shelf/runtime/parallel/parallel_topology.h`）が
座標、GPU 割当、グループを一元管理する。GPU 初期化や通信は行わない。

- `ParallelShape { pp, ep, tp }`
- `ParallelCoord { pp, ep, tp }`
- `rank(coord)` / `coord(rank)` / `device(rank)`
- `tp_group(pp, ep)` — 同じ EP グループ内の TP rank 列
- `ep_group(pp, tp)` — 同じ TP lane の EP rank 列
- `stage_group(pp)` — 1 pipeline stage の全 rank

group は Topology だけが計算する。`TpCoordinator` は受け取った `RankGroup` から
`TpTransport` と `TpBarrierGroup` を作るだけで、rank 座標を再計算しない。

`ExpertPartitionMode`:

| mode | 意味 |
| --- | --- |
| `Replicated` | expert 軸を分割しない |
| `AcrossEP` | routed expert を EP rank で分割する |
| `LegacyAcrossTP` | EP=1 の旧構成。expert 軸を TP rank に重ねる |

Topology 上は常に実際の `EP` を表す。EP=1 のときに expert 軸を TP に重ねるのは
partition policy であり、Topology の軸を崩さない。

## 2. Execution hierarchy

```text
ContinuousBatcher (global, 1 個)
      |
  ScheduledBatch
      |
  PipelineParallelCoordinator (PP>1)
    ├── stage 0: StageRuntime / layers [0, l1) / embedding
    ├── stage 1: StageRuntime / layers [l1, l2)
    └── stage K-1: StageRuntime / layers [lK-1, L) / final norm + lm_head
          |
        TpCoordinator (stage 内 EP x TP)
          ├── worker rank 0 ── TpRankRuntime (device-local Executor)
          └── worker rank N ── TpRankRuntime (device-local Executor)
```

- 1 rank = 1 device-local runtime。巨大な multi-device Executor は作らない。
- `PipelinePPCoordinator` は stage の順序と hidden hand-off、sampled token の返却を持つ。
- `TpCoordinator` は rank runtime 生成、同一 batch の配布、TP / EP reduce、
  rank failure の集約、rank 0 の結果返却を持つ。
- rank の model は `TpRankModel` interface 経由で注入する。runtime は checkpoint の
  family を知らない。builder の選択は `shelf::models::select_tp_rank_model` が行う。
- `GpuMemoryManager` が device ごとの `GpuArena` と `hipStream_t` を所有し、
  各 runtime は借りるだけである。runtime は arena / stream を生成・破棄しない
  （[runtime.md](runtime.md)）。

## 3. Weight partition

### matrix axis（TP）

`TensorPartitionPlan` が backbone と shared expert の matrix axis を分割する。
正本は [tensor_partition.md](tensor_partition.md)。

- dense MLP: gate/up は axis 0、down は axis 1
- full attention: q/k/v は axis 0、o は axis 1。GQA は q head と KV head の両方を割る
- GDN: in_proj / conv は axis 0、out_proj は axis 1。recurrent state は rank-local
- shared expert: gate/up は axis 0、down は axis 1
- lm_head: dense は vocab-parallel（BF16 / EXL3）。MoE の lm_head は replicated
- embedding / norm / router / `shared_expert_gate` は replicated

### expert axis（EP）

routed expert 平面は matrix axis を持たないため、`MoeExpertGroupPlan` が
expert 軸で切る。

```cpp
build_moe_expert_group_plan(config, layer, ep_size, ep_rank, tp_size, tp_rank);
```

- `experts * ep_size == global_experts`
- `expert_begin == ep_rank * experts`
- `tp_size` / `tp_rank` は shared expert の matrix axis 検証に使う
- 4 引数 overload は `ep_size = tp_size`, `ep_rank = tp_rank` の旧構成

`num_experts` が `ep_size` で割り切れない場合は `unsupported`。router は全 rank が
同じ logits と同じ top-k を持ち、所有範囲外の選択を `kMoeSlotNotOwned` にする。

## 4. Collectives

MoE の FFN は次の順で reduce する。

```text
routed expert partial
  → TP reduce（同一 EP グループ内）
  → EP reduce（TP lane ごとの代表 rank）
  → shared expert contribution を加算
```

- `tp_group(pp, ep)` が TP reduce の group。
- `ep_group(pp, tp)` が EP reduce の group。TP lane ごとに独立しており、同じ
  lane の全 EP rank が参加する。全 rank がちょうど 1 回ずつ EP barrier に参加する。
- shared expert は TP reduce にだけ参加し、EP reduce の対象にしない。
- reduction は f32 で累積し、最後に 1 回だけ bf16 へ丸める。
  peer 経路と host 経路は同じ contract を持つ。
- `HipPeerTpTransport` は peer write ベースの対称 push reduce。
  `HostMediatedTpTransport` は reference / fallback。
  `verify()` が失敗した場合は stderr に出して host 経路へ切り替える。
  `SHELF_TP_TRANSPORT=auto|p2p|host` で強制できる。
- RCCL は必須 dependency にしていない。all-to-all は未実装。

R9700 では kernel からの peer load が壊れるため、transport は peer store のみで
構成する。詳細は [../research/](../research/) の P2P transport 記録を参照する。

## 5. Pipeline parallel

- `--pipeline-parallel K` は `even_stage_ranges(L, K)` で layer を均等に割る。
- 先頭 stage が embedding、最終 stage が final norm / lm_head / sampling を持つ。
- stage 間は bf16 の hidden を pinned host 経由で直列に受け渡す。
- stage 内の EP / TP は stage ごとの `TpCoordinator` が持つ。PP + EP / TP を
  同時に指定できる。
- stage 間の hidden 幅は model の hidden と限らない。Qwen4Exp は
  HyperConnection の stream 全体（`hc_count * hidden_size`）を渡す。
  `ExecutorConfig::pipeline_hidden_width` が transfer buffer の幅を決める。
- 最終 stage の sampled token だけを stage 0 の device へ戻す。
- `PipelineStageRuntime` は arena / stream を所有しない。`GpuMemoryManager` から借りる。
- KV page 数は stage 0 の auto-fit を全 stage で共有する。block table の page ID を
  stage 間で一致させるためである。stage 0 より容量の小さい stage が同じ page 数を
  保持できない場合、`SHELF_KV_POOL_PAGES` で全 stage の page 数を明示する。

## 6. 比較基準

TP は部分和の reduce 順序が single device と変わるため、TP=2 の logits は single と
bit 同一にならない。token 列の完全一致を要求してよいのは、分割軸と reduce 順序が
同じ構成どうしだけである。

| 比較 | 判定 |
| --- | --- |
| single と PP=2 | token 列の完全一致（PP は層の配置だけを変える） |
| TP=2 と PP=2 + TP=2 | token 列の完全一致（同じ TP グループ、同じ reduce 順序） |
| single と TP=2 | 許容誤差。logits の相対誤差・top-1・必要なら KLD で判定し、token 列の完全一致を要求しない |
| TP=2 と EP=2 / EP=4 | expert の部分和の順序が変わるため許容誤差 |

TP=2 どうしでも、prompt の logits margin が小さい場合は丸め差で greedy token が
分岐し得る。E2E の採否は上表の組で行い、単一 GPU との token 一致を TP の
正しさの条件にしない。誤差の測り方は
[../perf/methodology.md](../perf/methodology.md) §10 に従う。

## 7. Numerical compatibility

partition された GEMM の出力（bf16）と barrier の SUM で丸めが入るため、
TP=1 と TP=2 は bit-identical ではない。Qwen3.8-27B FP8（64 層、
prefill 5 tokens / greedy 16 tokens）の layer ごとの相対 drift:

```text
layer  0: rel=0.6%
layer  3: rel=2.1%
layer  7: rel=2.2%
layer 15: rel=1.4%
layer 31: rel=1.3%
layer 62: rel=2.7%
final hidden: rel=4.4%   prefill logits max abs diff=0.70 (argmax 一致)
```

特定層での跳躍がなければ、partition / collective の bug ではなく bf16 丸めの累積と
判断する。rank 0 と rank 1 の final hidden / logits は SUM 以降 replicated なので
bit-identical である。

MXFP4 dense の TP=2 は同じ partition / preshuffle / GEMM 経路を使うが、W4A8 の
活性 quantization が rank-local の K 範囲を入力にするため、TP=1 と TP=2 で活性
plane の row reference が一致しない。MXFP4 の drift は FP8 と同じ帯に収まることを
要求しない。

## 8. 対応範囲

| model | PP | EP | TP |
| --- | --- | --- | --- |
| Qwen3.8-27B dense（MXFP4 / EXL3 / FP8） | 2 | — | 2 |
| Qwen3.8-27B QSA | 2 | — | 2 |
| Qwen3.8-Flash-Next（qwen4_exp MoE） | 2 | 2 / 4 | 2 |
| Qwen3.8-Flash-Next-DFlash2 drafter | — | — | 1 |

- EP は routed expert を持つ model だけに意味がある。dense は EP で分割せず、
  EP グループごとに同じ計算を複製する（高速化しない）。
- Qwen4Exp の TP=4 は KV head 数などの local geometry 条件で起動時に拒否される。
- `--tensor-parallel 1|2|4` は CLI で受け付ける。実際に起動できるかは model の
  head 数、intermediate width、lm_head の分割条件で決まる。
- dflash2 は drafter を分割せず、target verify のみ TP=2 で実行する。
- MTP block は partition しない。
- PP + prefix cache、PP + DFlash2、PP + MTP、PP + hidden taps、PP + `--dump-logits`
  は併用できない。
- decode と prefill が同じ batch に混在する実行は未対応。
- 異種 GPU は未サポート（同一 RDNA4 を前提）。

## 9. Tests

| test | label | 内容 |
| --- | --- | --- |
| `test_parallel_topology` | cpu;required | PP / EP / TP の座標変換と `tp_group` / `ep_group` / `stage_rank` |
| `test_qwen35_tp_context` | cpu;required | TP local geometry / schedule / collective boundary |
| `test_tp_reduction` | cpu;required | f32 accumulate、1 回だけの bf16 丸め |
| `test_qwen35_tensor_parallel_plan` | cpu;required | Qwen35 matrix partition plan |
| `test_tensor_partition` / `test_canonical_partition` | cpu;required | weight storage 側の partition |
| `test_weight_load` | gpu1;required | single-GPU 回帰 + partition plan なし経路 |
| `test_qwen35_tp_weight_load` | gpu1;required | rank-local weight load |
| `test_tp_transport` | gpu2;optional | `sum_hidden` の CPU reference oracle |
| `test_qwen35_tp_execution` | gpu2;optional | tiny model の TP 実行、rank 間 logits 一致、layer drift |
| `test_qwen35_tp_e2e` | gpu2;optional;external_files | 実 model の TP=1 / TP=2 greedy |
| `test_qwen35moe_tensor_parallel_plan` | cpu;required | MoE の rank plan（shared expert / router 非分割） |
| `test_qwen35moe_tp_execution` | gpu2;optional | MoE fixture の TP 実行と局所 geometry |
| `test_moe_router` | gpu1;required | routing / group plan / 非所有 slot |

GPU2 テストは `ctest -L gpu2` で実行する。GPU 不足時は exit 77 で skip し、required
acceptance には含まれない。

## 10. 非目標

- Multi-GPU Executor の一般化
- RCCL backend（AllReduce / AllGather / AllToAll）
- all-to-all expert dispatch
- 通信と計算の本格的な overlap
- TP schedule と HIP graph の併用
- distributed sampling / structured generation
