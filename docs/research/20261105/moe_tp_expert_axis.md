# MoE tensor parallel は expert 軸で分割し、routing は model 全体で行う

実験 ID `MOE-TP`、結論 **ACCEPT（expert 軸分割 + 全局 routing + combine で collect）**。

対象: `models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`（40 層・experts 256・top_k 8・
moe_intermediate 512・shared_expert_intermediate 512）を TP=2 で動かすための分割方針。
実測は合成 fixture（experts 4・layers 2）と実 model（Qwen3.6-35B-A3B）の TP=2 まで。
実 model で最初に破綻していた箇所と原因は §6 に残す。

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

## 5. 残作業

- `TpRankRuntime` の auto reserve（`qwen35_executor_reserve_bytes` + `TpRankModel::
  ffn_workspace_reserve_bytes`）は実 footprint より少ない（実測 gap: row bucket 16 で 0.01 GiB、
  bucket 192 で 0.13 GiB）。KV pool を残り領域で切る順序なので、過小評価は
  `arena capacity exceeded` の起動失敗になる。auto 経路には `max(64 MiB, free / 100)` の slack を
  残すようにした。容量式そのものの過小評価は残っている
- `main` の EXL3 系は execution workspace に 306〜816 MiB（row bucket 依存）を無条件で
  足していた。EXL3 linear を走らせる実行だけ確保する形へ変えたので、EXL3 を使わない model
  （BF16 / HF FP8 / MXFP4）の容量式は変更前と同じ値に戻る。経過と採否は
  [exl3_workspace_conditional.md](exl3_workspace_conditional.md)
- MoE pipeline parallel は未実装（順序は fp8 KV → MTP → TP → PP の最後）
- expert 配分は均等な contiguous 範囲のみ。負荷に応じた expert 配置（真の EP scheduler）は非目標
- MoE MTP block は分割しない（dense と同じ）

## 6. 実 model TP=2 の数値破綻（実行前処理の漏れ）

multi-rank runtime を配線して実 model を TP=2 で動かすと破綻した。原因と切り分けを残す。
contract 自体（expert 軸分割・global routing・combine で collect）は変わっていない。

### 症状（修正前）

`models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4`、PSKLDTOK corpus、teacher-forced 4 位置。

- prefill 自体は一致。ctx=192 の prefill 直後 sampled token は TP=1 / TP=2 rank0 / rank1 とも `24609`
- **履歴を読む最初の rows=1 step で壊れる**。ctx=192 / 256 は非有限値が router まで伝播して
  `status=6`（`NUMERIC_ERROR`）、ctx=64 は有限だが KL 11.09 と top-1 逆転、
  ctx=32 / 96 / 128 は KL ≤ 1.4e-2
- 壊れる context が非単調（64 壊れ、96 と 128 は無事、192 と 256 は非有限）

### 切り分け（`test_qwen35moe_tp_layer_trace`、ctx=192）

```bash
./build/tests/test_qwen35moe_tp_layer_trace \
  --model-dir models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 \
  --tokens-file /tmp/moe_corpus.psktok --context 192
```

TP=1 rank と TP=2 rank 0 / 1 を同一 token で走らせ、(1) prefill 直後の GDN conv / recurrent
state と KV pool を global 順序へ復元して比べ、(2) rows=1 step の value を DECODE program の
dispatch 一覧（kernel 名つき）と突き合わせて比べる。

1. **prefill 直後の内部状態**を rank-local から global channel / head 順序へ復元して TP=1 と比較。
   非有限値は無く、GDN conv rel ≤ 0.040、GDN recurrent rel ≤ 0.055、KV K rel 0.012 → 0.228
   （深い attention 層へ単調に増える）、KV V rel 0.028 → 0.151、position と page table は一致。
   state の書き込み・stride・page 割り当ては原因でないと分かった。KV の相対差が深い層で育つのは
   上流 hidden の漂移の累計と見る（この値自体は別の問題として残す）
2. **rows=1 step** を DECODE program の dispatch 一覧（kernel 名と value id）と突き合わせて比較。
   collect 後の hidden（layer 0 の `RESIDUAL_ADD` v20 / `RMS_NORM` v21 / router logits v22）は
   rel ≤ 5e-3 で一致。最初に壊れる値は **layer 0 の `MOE_COMBINE` 出力（v34）**で
   rank0 が `5.914e8`（non-finite 1 個）、rank1 が `-2.863e27`（non-finite 6 個）。
   collect 後の v35 では両 rank が同じ値（non-finite 7 個）になり、**部分和を書いた段階で既にゴミ**が
   入っている。以後 v36 以降が全 feature NaN となり、layer 1 の router logits（v56）で検知される
   → GDN state でも KV cache でもなく、layer 0 の routing plan 側

combine が読む row に rank 依存のゴミが出るのは、plane の残り値を読む signature と整合する。

### 原因

`MOE_ROUTER` は `tile_counts` へ **累算**する（`atomicAdd`）ので、routing round は zero から
始めなければならない。その zeroing は `prepare_program_run` が担うが、program run の開始点が
TP 経路と non-TP 経路で分かれており、**TP 経路だけが呼んでいなかった**。dense TP では
`plan_scratch_slot == kNoWorkspace` で no-op なので、grouped expert を multi rank で走らせて
初めて露出した。

`tile_counts` が前回の値から累算されると（workspace のこの byte は weight load の staging が
使った領域であり zero とは限らない）、`moe_slot_scan` の `expert_row_begin` / `tile_base` /
group table が狂い、`moe_slot_scatter` が割り当てる row と grouped GEMM が書く row の集合が
食い違い、`MOE_COMBINE` が未書き込みの byte を読む。rank でゴミが違う・context で壊れ方が
非単調・合成 fixture は通る（残っていた byte が無害）という観測すべてと整合する。

### 修正

program run の開始点を `launch_program_run`（`src/shelf/models/qwen35/runtime/executor.hip`）
1 本に寄せ、実行経路を選ぶ前に `prepare_program_run` → error word memset を行う形にした。
経路側（`execute_program_range` / `execute_program_tp_ranges`）は前処理しない。前処理を経路の
内側に置くと、将来経路が増えたとき同じ漏れが再発する。

回帰は `test_qwen35moe_tp_execution` が固定する。全 program の plan scratch を `0xff` で埋めた
状態で prefill / decode を走らせ、TP=1 と TP=2 の greedy token が一致することで見る
（前処理を飛ばす経路では参考結果に到達できない）。

### 修正後の実測（TP=1 vs TP=2、teacher-forced 4 位置、同一 corpus）

| context | TP=2 status | 行ごとの最大 KL | top-1 一致 |
| --- | --- | ---: | --- |
| 32 | ok | 2.6e-4 | 4/4 |
| 64 | ok | 1.3e-2 | 3/4（near-tie 逆転 1） |
| 96 | ok | 8.2e-2 | 3/4 |
| 128 | ok | 1.3e-2 | 4/4 |
| 192 | ok | 2.6e-4 | 4/4 |
| 256 | ok | 3.5e-2 | 3/4 |

同じ corpus で TP=1 の prefill chunk を変えたときの KL 幅が 1e-4 〜 1.4e-2
（[moe_mtp.md](moe_mtp.md)）なので、大半は reduction 順序の差の帯に入っている。
ctx=96 の 8.2e-2 だけは帯より広く、TP=1 側の chunk 感度ともう一度突き合わせる価値がある（未追跡）。

## 7. TP=2 の backbone を registered 側へ載せる

数値が一致するようになった直後、TP=2 の prefill が TP=1 の 1/25 であった。原因は rank local
shape が registered 側の allowlist に無く backbone が correctness path へ落ちていたこと。
現在の契約（登録済み shape）は [../../developer/tensor_parallel_execution.md](../../developer/tensor_parallel_execution.md) §11-1 に置く。

### 切り分け

`SHELF_OP_DUMP=1` で kernel 合計を TP=1 と TP=2 の pp（prompt 512・chunk 512・1 run）で比べた。
TP=2 は 2 rank 合計なので per-rank に見る倍率は表の 1/2 である。registered 側は per-rank で 1 を
切る（grouped は expert 数が半分になるので 0.7 程度）ので、2.5 倍を超えている kernel が
fallback と判断できた。

| kernel | TP=1 ms | TP=2 ms | 倍率 |
| --- | ---: | ---: | ---: |
| `PAGED_ATTENTION` | 1.71 | 734.25 | 428 |
| `GDN_RECURRENCE` | 7.97 | 1264.53 | 159 |
| `LINEAR_BF16` | 88.34 | 2526.20 | 28.6 |
| `RMS_NORM` | 8.20 | 134.99 | 16.5 |
| `RESIDUAL_ADD` | 2.73 | 34.97 | 12.8 |
| `ROPE` | 0.89 | 11.59 | 13.1 |
| `SPLIT` | 0.40 | 4.73 | 11.8 |
| `MUL` | 1.79 | 19.17 | 10.7 |
| `SILU` | 3.27 | 28.78 | 8.8 |
| `STATEFUL_CAUSAL_CONV1D` | 2.73 | 22.68 | 8.3 |
| `KV_APPEND` | 1.06 | 6.16 | 5.8 |
| `SWIGLU` | 3.60 | 21.07 | 5.8 |
| registered だったもの（`LINEAR_MXFP4_W4A8_GROUPED` 27.34/38.40、`ACTIVATION_QUANTIZE_MXFP4_ROWREF` 6.75/11.36、`MOE_ROUTER` 2.04/4.42、`MOE_COMBINE` 4.46/6.60、`MOE_PLAN_SCAN` 3.09/4.40） | 43.68 | 65.18 | 1.40〜2.17 |

`RESIDUAL_ADD` の 12.8 倍（per-rank 6.4 倍）は fallback では無い。collect の直後の op で、
peer への send と sum の待りが次の op の計測区間に載るため見える。barrier の費用を kernel 倍率
で読むと誤診する。

どの dispatch が落ちているかは `SHELF_OP_FALLBACK_DEBUG=1`（op index、kernel、io の feature
count・dtype・row domain を出す）で特定した。`SHELF_QWEN35_PERF_STATS` の kernel 別 counter は
rank thread で集計されず main thread の dump は 0 になる（multi rank では `SHELF_OP_DUMP` か
`SHELF_OP_FALLBACK_DEBUG` を使う）。

### 登録した rank local shape

- `LINEAR_BF16`: (2048,2048)（in_proj_z / out_proj / o_proj）、(2048,256)（shared expert down）、
  (16,2048)（GDN in_proj_a / in_proj_b）。shared gate / up と k_proj の (256,2048)、
  in_proj_qkv と q_proj の (4096,2048) は元から登録済み
- `STATEFUL_CAUSAL_CONV1D`: conv features 4096
- `GDN_RECURRENCE`: (key heads 8, value heads 16)
- `PAGED_ATTENTION`: (q 8, kv 1) を bf16 / fp8 双方。`q_per_kv` は 8 のままなので visible 上限は
  (16,2) と同じ model の max sequence length
- `KV_APPEND`: local KV head 1 を bf16 / fp8 双方
- `ROPE`: features 2048（local query）と 256（local key）
- `RMS_NORM`: GDN head norm（F32→F32, features 2048, group 128）、per-head QK norm
  （PerGroup (2048,256) と PerFeature (256,256)）
- elementwise: `SiluF32ToBf16` 4096、`SiluBf16ToF32` 2048、`MulF32F32ToBf16` 2048、
  `SwigluBf16` 256、`SplitBf16InterleavedHeads` (2048,256)、`SigmoidBf16ToF32` 2048

selector の unit test には正例として入れた。今まで negative case（「未登録」）に使っていた
shape（`gdn_conv` の conv_dim 4096、`gdn_recurrence` の heads (8,16)、elementwise の
`SiluF32ToBf16` 4096）は、未登録の側へ移し替えた。

### dispatch 側で見つけた 2 つの落とし穴

1. **per-head norm の weight layout は rank の平面幅で変わる。** local KV head が 1 の rank では
   K 平面の幅が `head_dim` と同じになり、`resolve_rmsnorm_physical` は
   `slot.elements == F`（PER_FEATURE）を PER_GROUP より先に見るので PerFeature として selector へ
   渡る。allowlist には PerGroup 版（TP=1 の (4096,256)・(512,256)）しか無く、PerFeature 版
   (256,256) が無かった。PER_FEATURE の方が意味として正しい（1 head の平面に weight が 1 本ずつ
   対応する）
2. **`shelf-bench pp --mode forward` が logits を計算していなかった**（`compute_logits` が
   `mode == greedy || tp != nullptr`）。TP=1 の pp では program 内の lm_head 系 4 dispatch
   （`OUTPUT_GATHER` → 最終 `RMS_NORM` → lm_head `LINEAR_BF16` → `SAMPLING`）が
   `ctx.actual_outputs == 0` で NotApplicable になり、「TP=1 だけ fallback 4 本」という見かけに
   なっていた。実費は 0 row の no-op だが、TP=1 と TP=2 で dispatch される op 集合が変わるので
   誤解を招く。pp は mode にかかわらず最終 chunk の最終行の logits を計算するようにした

### 登録後

pp prompt 512・chunk 512・1 run の `SHELF_OP_FALLBACK_DEBUG` 行数は TP=1 **0** / TP=2 **0**
（bf16 KV。fp8 KV も TP=1 **0** / TP=2 **0**）。

prefill throughput（`shelf-bench pp --prompt-tokens 8192 --runs 3 --warmup 1`、median）:

| prefill chunk | TP=1 bf16 | TP=2 bf16 登録前 | TP=2 bf16 登録後 |
| ---: | ---: | ---: | ---: |
| 1024 | 3777.7 | 123.4 | 6209.5 |
| 2048 | 3785.0 | 152.4 | 6543.6 |
| 4096 | 3616.7 | 151.6 | 6466.5 |

fp8 KV（chunk 2048）は TP=1 3743.3 / TP=2 6491.3。expert 平面を rank で分けるので weight 読みが
半分になり、DRAM 律速の prefill では TP=2 が素直に速い（1.64〜1.79 倍）。性能値の正本は
[../../perf/current.md](../../perf/current.md)。

optimized kernel を初めて通す shape があるため、selector 変更後の teacher-forced 再測
（TP=1 vs TP=2）がそのまま A/B になる。値は §6 の表と同じ帯であることを確認し、
`../../perf/current.md` に置く。
