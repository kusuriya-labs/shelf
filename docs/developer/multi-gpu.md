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
| `Replicated` | expert 軸を分割しない。全 expert を各 TP rank に置く |
| `AcrossEP` | routed expert を EP rank で分割する |

EP は expert ID の軸、TP は matrix 軸である。EP=1 では全 expert を各 TP rank に
複製し、TP が各 expert の gate/up の N と down の K を分割する。EP>1 では
EP が expert ID を分割した後、同じ TP が各 local expert の行列を分割する。
router は全 rank が同じ top-k を選ぶため replicated のままである。

EP>1 かつ TP=1 で、attention / GDN の head 数が EP で割り切れる場合、EP rank を
backbone の TP 軸にも使う（axes overlap）。backbone、shared expert、lm_head は
EP グループ内で TP 分割され、routed expert は EP 軸だけで分割される。
MTP block も `--serve-stdio --mtp` では同じ backbone TP 軸で分割する。
`num_key_value_heads` などが EP で割り切れない場合は backbone を複製する
fallback 経路になる。Qwen3.8-Flash-Next EXL3（KV head 2）は EP=2 で overlap し、
EP=4 では KV head が 4 で割れないため fallback する。
axes overlap の backbone 分割を実装しているのは現在 qwen4exp である。
qwen35moe の EP 単独構成は backbone を複製する。

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

`TensorPartitionPlan` が backbone と（qwen35moe の）shared expert の matrix axis を
分割する。正本は [tensor_partition.md](tensor_partition.md)。

- dense MLP: gate/up は axis 0、down は axis 1
- full attention: q/k/v は axis 0、o は axis 1。GQA は q head と KV head の両方を割る
- GDN: in_proj / conv は axis 0、out_proj は axis 1。recurrent state は rank-local
- routed expert: gate/up は N 軸（axis 0）、down は K 軸（axis 1）
- qwen35moe の shared expert: gate/up は axis 0、down は axis 1
- qwen4exp の shared expert: replicated（matrix 分割しない）
- lm_head: dense は vocab-parallel。qwen35moe は replicated。qwen4exp は
  checkpoint の量子化形式が許す場合に vocab-parallel
- embedding / norm / router / `shared_expert_gate` は replicated

### expert axis（EP）

`MoeExpertGroupPlan` が expert ID を切り、同じ plan の `tp_size` / `tp_rank` が
各 expert の matrix 軸を切る。

```cpp
build_moe_expert_group_plan(config, layer, ep_size, ep_rank, tp_size, tp_rank);
```

- `experts * ep_size == global_experts`
- `expert_begin == ep_rank * experts`
- `w13_gate_rows` / `w13_up_rows` は `moe_intermediate_size / tp_size`
- `w2_cols` は `moe_intermediate_size / tp_size`
- 4 引数 overload は `ep_size = 1`, `ep_rank = 0` の TP-only 構成

`num_experts` が `ep_size` で割り切れない場合は `unsupported`。router は全 rank が
同じ logits と同じ top-k を持ち、所有範囲外の選択を `kMoeSlotNotOwned` にする。
MXFP4 / EXL3 の expert を TP 分割するときは、各 rank の N / K が量子化形式の
alignment を満たす必要がある。EXL3 INT8 は N と K の双方が 128 の倍数である
必要があり、`moe_intermediate_size / (128 * tp_size)` が整数でない構成は
load 時に `unsupported` になる。

## 4. Collectives

MoE の FFN は次の順で reduce する。

```text
routed expert partial
  → TP reduce（同一 EP グループ内）
  → EP reduce（TP lane ごとの代表 rank）
  → shared expert contribution を加算
```

- `tp_group(pp, ep)` が TP reduce の group。EP=1 では全 expert を複製しているため、
  この reduce が routed expert の matrix tile を足し合わせる。
- `ep_group(pp, tp)` が EP reduce の group。TP lane ごとに独立しており、同じ
  lane の全 EP rank が参加する。全 rank がちょうど 1 回ずつ EP barrier に参加する。
  EP=1 では EP reduce は存在しない。
- axes overlap（EP>1, TP=1）では EP group が backbone の TP group も兼ねる。
  backbone の `tp_combine` は EP group で reduce する。qwen4exp では routed と shared の
  両方を含む 1 個の MoE combine に `ep_combine` を付け、EP group で 1 回だけ reduce
  する。これにより routed 用と shared 用の 2 barrier を 1 barrier に畳む。
- shared expert は通常の EP reduce の対象にしない。qwen4exp の shared expert は
  replicated で、backbone TP rank 0 だけが寄与する。
- reduction は f32 で累積し、最後に 1 回だけ bf16 へ丸める。
  peer 経路と host 経路は同じ contract を持つ。
- `HipPeerTpTransport` は peer write ベースの対称 push reduce。
  `HostMediatedTpTransport` は reference / fallback。
  `verify()` が失敗した場合は stderr に出して host 経路へ切り替える。
  `SHELF_TP_TRANSPORT=auto|p2p|host` で強制できる。
- RCCL は必須 dependency にしていない。all-to-all は未実装。

R9700 では kernel からの peer load が壊れるため、transport は peer store のみで
構成する。詳細は [../research/](../research/) の P2P transport 記録を参照する。

### TP inbox の事前確保

TP の inbox は `TpCoordinator::initialize` で確保する。推論中の `hipMalloc` は行わない。

- `tp_inbox_bytes()` が全 program / TP schedule の combine 値から最大 byte 数を求める。
  起動時に `TP_WORKSPACE inbox_bytes=... groups=...` を stderr に出す。
- `TpTransport::reserve(bytes)` が `verify()` の後、worker 起動前に inbox を確保する。
  `fused barrier` の host flag と arrival counter も同じ時点で確保する。
- 実行中の `sum_hidden` / `sum_hidden_fused` / `peer_write_target` は容量を検査するだけで、
  確保も拡張もしない。容量不足は `Status::unsupported` で返す。
- `TpCoordinator::attach_aux_executor` は drafter の program から必要量を再計算し、
  worker 起動前に reserve する。

### Fused barrier（2 rank、opt-in）

`SHELF_TP_FUSED_BARRIER=1` では、`TpCombineFusion::None` と `ResidualAdd` の combine を
1 kernel で実行する。

1. 自分の部分和を peer の inbox へ copy
2. system scope の fence と arrival counter
3. peer の host-mapped flag を spin
4. inbox を読んで sum / residual add

- host thread は barrier で rendezvous しない。`TpBarrierGroup::arrive` は kernel を
  enqueue して返る。
- `HyperApply` と、inbox 容量を超える combine は CPU barrier 経路を使う。
- 対象は 2 rank のみである。3 rank 以上では `fused_barrier_ready()` が false になる。
- flag は host-mapped memory に置く。device memory の flag は peer から見えない。

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
| single と TP=2 | 許容誤差。routed expert も matrix 分割されるため、logits の相対誤差・top-1・必要なら KLD で判定し、token 列の完全一致を要求しない |
| single と EP=2 | 許容誤差。expert ID の部分和の順序が変わる |
| TP=2 と EP=2 + TP=2 | 許容誤差。expert の group と reduce 順序が変わる |
| EP=2 と EP=2 + TP=2 | 許容誤差。expert の matrix 分割が加わる |

TP=2 どうしでも、prompt の logits margin が小さい場合は丸め差で greedy token が
分岐し得る。E2E の採否は上表の組で行い、単一 GPU との token 一致を TP の
正しさの条件にしない。誤差の測り方は
[../perf/methodology.md](../perf/methodology.md) §10 に従う。

## 7. Numerical compatibility

partition された GEMM の出力（bf16）と barrier の SUM で丸めが入るため、
TP=1 と TP=2 は bit-identical ではない。MoE の routed expert も TP で matrix 分割
される場合は、combine の TP reduce で同じ丸めが入る。Qwen3.8-27B FP8（64 層、
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
- EXL3 INT8 の routed expert は N と K が 128 の倍数でなければならない。
  `moe_intermediate_size / (128 * tp_size)` が整数でない TP は load 時に
  `unsupported` になる。Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw（intermediate 640）は
  TP=2 でこの条件を満たさない。
- qwen35moe の MXFP4 routed expert も local の K が native group alignment（128）を
  満たす必要がある。
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
| `test_qwen35moe_tensor_parallel_plan` | cpu;required | MoE の rank plan（routed expert の matrix TP / shared expert / router 非分割） |
| `test_qwen35moe_tp_execution` | gpu2;optional | MoE fixture の TP 実行、routed expert の N/K 分割、局所 geometry |
| `test_qwen4exp_load` | gpu2;optional;external_files | EXL3 expert TP の 128 alignment 条件。満たさない checkpoint は load 時に拒否される |
| `test_moe_router` | gpu1;required | routing / group plan / 非所有 slot |

GPU2 テストは `ctest -L gpu2` で実行する。GPU 不足時は exit 77 で skip し、required
acceptance には含まれない。

## 10. TP / EP decode の segment graph

`tp_schedules` を持つ Executor でも、decode の compute 区間は HIP graph に capture される。
単位は program 全体ではなく `TpExecutionSchedule` の segment で、collective は従来どおり
CPU 側の `arrive()` が担う。

- cache は `Executor::tp_segment_graphs`（program index × segment index）。
- capture は staging pool が warm になった後の run から。`ctx.capturing_graph` を立てる。
- node が 0 の segment は capture せず eager で実行する。
- 値が変わり得る入力（row 数 / request 数 / device pointer 群）を key にし、
  一致しない run は eager に落とす。既定は無効で、`SHELF_TP_HIP_GRAPH=1` で有効化する。
- 対象は `ExecutionRole::Decode` の run。`SHELF_TP_HIP_GRAPH_VERIFY=1` で
  `ExecutionRole::Verify` も対象になる（既定 off）。prefill は常に eager。
- 実行方針は `TpGraphPolicy` の 3 値。
  - `Disabled`: 常に eager。非 decode、`SHELF_TP_HIP_GRAPH=0`、graph 失敗時。
  - `ReplayOnly`: 登録済み graph のみ replay し、未登録なら eager。DFlash2 の単発 target
    decode (`run_single_target`) が `ExecuteBatchOptions::tp_graph_replay_only` で指定する。
  - `CaptureAndReplay`: 未登録なら capture。通常の decode。
- capture は同じ key の run が 2 回続いた後にだけ許可し、1 run あたりの登録数を
  `SHELF_TP_GRAPH_CAPTURE_BUDGET`（既定 8）に制限する。dispatch 数の多い segment から
  登録するので、途中の run でも効き始める。
- key は role / 行数 / request 数 / slot の device pointer / tree metadata /
  GDN history / QSA history / PLE conv state と、program 内の attention plan 署名
  (`paged_attention_plan_signature` と QSA select の分岐) から作る。context 長そのものは
  含めない。含めると毎 round 変わり replay できないため。
- PLE の external input は graph の外で毎 step upload されるので、`external_input_preparer`
  を持っていても section graph は使える。
- collective（`tp_copy` / `tp_sum_*`）と zero fill は graph の外に残る。graph 内へ
  閉じ込めるには cross-device 同期を kernel で表現する必要があり、現行 transport では
  可視性を保証できない（`docs/research/20261124/tp_hip_graph.md`）。

graph は GPU の kernel 数を減らさない。消えるのは host 側の投入待ちで生じる GPU idle で
ある。dispatch 数そのものを減らすには kernel 融合が別途必要になる。

## 11. 非目標

- Multi-GPU Executor の一般化
- RCCL backend（AllReduce / AllGather / AllToAll）
- all-to-all expert dispatch
- 通信と計算の本格的な overlap
- collective を graph 内に閉じ込めた full decode graph
- distributed sampling / structured generation
